"""Shared instrumentation for the reference runner and the port: env, timing, memory, RNG record/replay, dumps.

Nothing here changes the arithmetic of the code it wraps: timers synchronise the device around a call, the RNG
recorder returns the very tensor the framework produced (record) or a stored tensor of identical dtype/shape
(replay), and dump hooks copy outputs to the host after they were computed.
"""
import contextlib
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from collections import OrderedDict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_REPO = os.path.abspath(os.path.join(ROOT, '..', 'reference', 'idol-sdk'))

IMAGES = {
    'robot': (os.path.join(REF_REPO, 'demo', 'examples', 'fig-robot-cute.webp'), 'synthetic'),
    'avocado': (os.path.join(REF_REPO, 'demo', 'examples', 'food-avocado.webp'), 'real'),
}


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# -- memory --------------------------------------------------------------------------------------------------

class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [('ri_uuid', ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        'ri_user_time', 'ri_system_time', 'ri_pkg_idle_wkups', 'ri_interrupt_wkups', 'ri_pageins', 'ri_wired_size',
        'ri_resident_size', 'ri_phys_footprint', 'ri_proc_start_abstime', 'ri_proc_exit_abstime',
        'ri_child_user_time', 'ri_child_system_time', 'ri_child_pkg_idle_wkups', 'ri_child_interrupt_wkups',
        'ri_child_pageins', 'ri_child_elapsed_abstime', 'ri_diskio_bytesread', 'ri_diskio_byteswritten',
        'ri_cpu_time_qos_default', 'ri_cpu_time_qos_maintenance', 'ri_cpu_time_qos_background',
        'ri_cpu_time_qos_utility', 'ri_cpu_time_qos_legacy', 'ri_cpu_time_qos_user_initiated',
        'ri_cpu_time_qos_user_interactive', 'ri_billed_system_time', 'ri_serviced_system_time',
        'ri_logical_writes', 'ri_lifetime_max_phys_footprint', 'ri_instructions', 'ri_cycles', 'ri_billed_energy',
        'ri_serviced_energy', 'ri_interval_max_phys_footprint', 'ri_runnable_time')]


_libproc = ctypes.CDLL('/usr/lib/libproc.dylib') if platform.system() == 'Darwin' else None


def footprint():
    """(current phys_footprint, lifetime max phys_footprint) of this process in bytes (unified memory: includes
    Metal buffers)."""
    info = _RUsageInfoV4()
    if _libproc is None or _libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return 0, 0
    return info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint


class MemorySampler:
    """Polls the MPS driver allocation and the process footprint; peak() since the last reset()."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self._stop = threading.Event()
        self.reset()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reset(self):
        self.peak_driver = 0
        self.peak_footprint = 0
        self.peak_allocated = 0

    def _sample(self):
        try:
            self.peak_driver = max(self.peak_driver, torch.mps.driver_allocated_memory())
            self.peak_allocated = max(self.peak_allocated, torch.mps.current_allocated_memory())
        except Exception:
            pass
        self.peak_footprint = max(self.peak_footprint, footprint()[0])

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def peak(self):
        self._sample()
        return {'mps_driver_peak_gb': self.peak_driver / 2**30, 'mps_allocated_peak_gb': self.peak_allocated / 2**30,
                'footprint_peak_gb': self.peak_footprint / 2**30}

    def stop(self):
        self._stop.set()


# -- timing -----------------------------------------------------------------------------------------------------

class StageTimer:
    """Accumulates synchronised wall time per stage name (nested names allowed)."""

    def __init__(self):
        self.times = OrderedDict()

    @contextlib.contextmanager
    def __call__(self, name):
        sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync()
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t0

    def wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            with self(name):
                return fn(*args, **kwargs)
        wrapped.__wrapped__ = fn
        return wrapped


# -- RNG record / replay ----------------------------------------------------------------------------------------

_RNG_FUNCS = ('randn', 'rand', 'randn_like', 'rand_like')


class RNG:
    """Intercepts torch.randn/rand/randn_like/rand_like. mode 'record': call through and keep a host copy of every
    result in order; mode 'replay': return the stored tensors in order (checked for shape/dtype), so a build can run
    on exactly the reference's random numbers without reproducing the generator."""

    def __init__(self, mode='record', stored=None):
        assert mode in ('record', 'replay')
        self.mode = mode
        self.stored = stored or []
        self.records = []
        self._orig = {}
        self.index = 0

    def _make(self, name):
        orig = self._orig[name]

        def fn(*args, **kwargs):
            if self.mode == 'record':
                out = orig(*args, **kwargs)
                self.records.append((name, out.detach().cpu().clone()))
                return out
            ref = orig(*args, **kwargs)   # same allocation, device and dtype as the reference call
            if self.index >= len(self.stored):
                raise RuntimeError(f'RNG replay: call {self.index} ({name}) beyond the recorded stream')
            rname, val = self.stored[self.index]
            self.index += 1
            if rname != name or tuple(val.shape) != tuple(ref.shape) or val.dtype != ref.dtype:
                raise RuntimeError(f'RNG replay mismatch at call {self.index - 1}: recorded {rname} '
                                   f'{tuple(val.shape)} {val.dtype}, got {name} {tuple(ref.shape)} {ref.dtype}')
            ref.copy_(val)
            return ref
        return fn

    def __enter__(self):
        for name in _RNG_FUNCS:
            self._orig[name] = getattr(torch, name)
        for name in _RNG_FUNCS:
            setattr(torch, name, self._make(name))
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(torch, name, fn)

    def as_dict(self):
        return {f'rng/{i:04d}_{name}': t for i, (name, t) in enumerate(self.records)}

    @staticmethod
    def from_dict(d):
        keys = sorted(k for k in d if k.startswith('rng/'))
        return [(k.split('_', 1)[1], d[k]) for k in keys]


# -- dumps ------------------------------------------------------------------------------------------------------

class Dump:
    """Ordered name -> host tensor store; names made unique with a #n suffix."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.data = OrderedDict()
        self.aliases = OrderedDict()   # name -> name of an identical (same live tensor object) entry
        self._seen = {}

    def put(self, name, value):
        if not self.enabled:
            return
        if isinstance(value, torch.Tensor):
            import weakref
            ident = (id(value), value._version)
            seen = self._seen.get(ident)
            if seen is not None and seen[0]() is value:
                key, n = name, 1
                while key in self.data or key in self.aliases:
                    n += 1
                    key = f'{name}#{n}'
                self.aliases[key] = seen[1]
                return
            value_ref = weakref.ref(value)
            orig_value = value
            value = value.detach()
            if value.is_complex():
                value = torch.view_as_real(value)
            value = value.cpu().contiguous().clone()
        else:
            value = torch.as_tensor(value)
        key, n = name, 1
        while key in self.data or key in self.aliases:
            n += 1
            key = f'{name}#{n}'
        self.data[key] = value
        if 'value_ref' in locals():
            self._seen[(id(orig_value), orig_value._version)] = (value_ref, key)

    def save(self, path, metadata=None):
        from safetensors.torch import save_file
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        meta = {k: str(v) for k, v in (metadata or {}).items()}
        meta['aliases'] = json.dumps(self.aliases)
        save_file(dict(self.data), path, metadata=meta)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def environment():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ''
    import numpy
    import PIL
    info = {
        'machine': run(['sysctl', '-n', 'machdep.cpu.brand_string']),
        'memory_gb': int(run(['sysctl', '-n', 'hw.memsize']) or 0) / 2**30,
        'macos': platform.mac_ver()[0],
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': numpy.__version__,
        'pillow': PIL.__version__,
        'reference_commit': run(['git', '-C', REF_REPO, 'rev-parse', 'HEAD']),
        'port_commit': run(['git', '-C', ROOT, 'rev-parse', 'HEAD']),
    }
    for mod in ('transformers', 'diffusers', 'tensordict', 'torchvision'):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


class DumpReader:
    """Read a dump written by Dump.save, resolving aliases (entries recorded as 'same tensor as <key>')."""

    def __init__(self, path):
        from safetensors import safe_open
        self.f = safe_open(path, 'pt')
        meta = self.f.metadata() or {}
        self.meta = meta
        self.aliases = json.loads(meta.get('aliases', '{}'))
        self.keys = set(self.f.keys()) | set(self.aliases)

    def __contains__(self, k):
        return k in self.keys

    def get(self, k, device=None):
        while k in self.aliases:
            k = self.aliases[k]
        t = self.f.get_tensor(k)
        return t.to(device) if device is not None else t

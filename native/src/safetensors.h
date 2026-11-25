// Minimal read-only safetensors reader (mmap). Header JSON is parsed for the subset the format uses:
// {"name": {"dtype": "F16", "shape": [..], "data_offsets": [b, e]}, "__metadata__": {"k": "v", ...}}
#pragma once
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace st {

struct TensorInfo {
  std::string dtype;            // F32, F16, BF16, I64, I32, U8, BOOL, F64
  std::vector<int64_t> shape;
  size_t begin = 0, end = 0;     // byte offsets into the data section
  const uint8_t* data = nullptr; // pointer into the mapping
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  size_t nbytes() const { return end - begin; }
};

inline size_t dtype_size(const std::string& d) {
  if (d == "F64" || d == "I64") return 8;
  if (d == "F32" || d == "I32") return 4;
  if (d == "F16" || d == "BF16" || d == "I16") return 2;
  if (d == "U8" || d == "I8" || d == "BOOL") return 1;
  throw std::runtime_error("safetensors: unknown dtype " + d);
}

class File {
 public:
  explicit File(const std::string& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("safetensors: cannot open " + path);
    struct stat sb;
    fstat(fd_, &sb);
    size_ = sb.st_size;
    map_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) throw std::runtime_error("safetensors: mmap failed " + path);
    uint64_t hlen;
    std::memcpy(&hlen, map_, 8);
    std::string header((const char*)map_ + 8, hlen);
    const uint8_t* base = map_ + 8 + hlen;
    size_t pos = 0;
    parse_top(header, pos, base);
  }
  ~File() {
    if (map_) munmap((void*)map_, size_);
    if (fd_ >= 0) ::close(fd_);
  }
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  bool has(const std::string& k) const { return tensors_.count(k) > 0; }
  // Dumps written by tools/common.py store repeated tensors once and list the others in an "aliases" metadata
  // entry (a flat JSON object name -> stored name); resolve those transparently.
  const TensorInfo& get(const std::string& k0) const {
    std::string k = k0;
    if (!tensors_.count(k)) {
      auto a = aliases().find(k);
      if (a != aliases().end()) k = a->second;
    }
    auto it = tensors_.find(k);
    if (it == tensors_.end()) throw std::runtime_error("safetensors: missing tensor '" + k + "' in " + path_);
    return it->second;
  }
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  const std::map<std::string, std::string>& metadata() const { return meta_; }
  const std::map<std::string, std::string>& aliases() const {
    if (!aliases_parsed_) {
      aliases_parsed_ = true;
      auto it = meta_.find("aliases");
      if (it != meta_.end() && it->second.size() > 2) {
        const std::string& s = it->second;
        size_t p = 0;
        expect(s, p, '{');
        while (true) {
          ws(s, p);
          if (s[p] == '}') break;
          std::string a = str(s, p);
          expect(s, p, ':');
          aliases_[a] = str(s, p);
          ws(s, p);
          if (s[p] == ',') { p++; continue; }
          break;
        }
      }
    }
    return aliases_;
  }

 private:
  static void ws(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\t' || s[p] == '\r')) p++;
  }
  static void expect(const std::string& s, size_t& p, char c) {
    ws(s, p);
    if (p >= s.size() || s[p] != c) throw std::runtime_error(std::string("safetensors: header parse error, expected ") + c);
    p++;
  }
  static std::string str(const std::string& s, size_t& p) {
    ws(s, p);
    expect(s, p, '"');
    std::string out;
    while (p < s.size() && s[p] != '"') {
      if (s[p] == '\\') {
        p++;
        char e = s[p];
        if (e == 'n') out += '\n';
        else if (e == 't') out += '\t';
        else if (e == 'u') {  // keep \uXXXX escapes verbatim (only seen in metadata)
          out += "\\u";
          out += s.substr(p + 1, 4);
          p += 4;
        } else out += e;
        p++;
      } else out += s[p++];
    }
    p++;
    return out;
  }
  static int64_t num(const std::string& s, size_t& p) {
    ws(s, p);
    size_t q = p;
    while (q < s.size() && (s[q] == '-' || (s[q] >= '0' && s[q] <= '9'))) q++;
    int64_t v = std::stoll(s.substr(p, q - p));
    p = q;
    return v;
  }
  static std::vector<int64_t> ints(const std::string& s, size_t& p) {
    std::vector<int64_t> v;
    expect(s, p, '[');
    ws(s, p);
    if (s[p] == ']') { p++; return v; }
    while (true) {
      v.push_back(num(s, p));
      ws(s, p);
      if (s[p] == ',') { p++; continue; }
      expect(s, p, ']');
      return v;
    }
  }
  void parse_top(const std::string& s, size_t& p, const uint8_t* base) {
    expect(s, p, '{');
    ws(s, p);
    if (s[p] == '}') return;
    while (true) {
      std::string key = str(s, p);
      expect(s, p, ':');
      if (key == "__metadata__") {
        expect(s, p, '{');
        ws(s, p);
        if (s[p] != '}') {
          while (true) {
            std::string k = str(s, p);
            expect(s, p, ':');
            meta_[k] = str(s, p);
            ws(s, p);
            if (s[p] == ',') { p++; continue; }
            break;
          }
        }
        expect(s, p, '}');
      } else {
        TensorInfo t;
        expect(s, p, '{');
        while (true) {
          std::string f = str(s, p);
          expect(s, p, ':');
          if (f == "dtype") t.dtype = str(s, p);
          else if (f == "shape") t.shape = ints(s, p);
          else if (f == "data_offsets") {
            auto o = ints(s, p);
            t.begin = o.at(0);
            t.end = o.at(1);
          } else throw std::runtime_error("safetensors: unexpected field " + f);
          ws(s, p);
          if (s[p] == ',') { p++; continue; }
          expect(s, p, '}');
          break;
        }
        t.data = base + t.begin;
        tensors_[key] = t;
      }
      ws(s, p);
      if (s[p] == ',') { p++; continue; }
      expect(s, p, '}');
      break;
    }
  }

  std::string path_;
  int fd_ = -1;
  size_t size_ = 0;
  const uint8_t* map_ = nullptr;
  std::map<std::string, TensorInfo> tensors_;
  std::map<std::string, std::string> meta_;
  mutable bool aliases_parsed_ = false;
  mutable std::map<std::string, std::string> aliases_;
};

}  // namespace st

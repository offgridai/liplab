#pragma once

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using int8 = std::int8_t;
using uint8 = std::uint8_t;
using int16 = std::int16_t;
using uint16 = std::uint16_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;
using int64 = std::int64_t;
using uint64 = std::uint64_t;
using TCHAR = char;

#define TEXT(x) x
#define OFFGRIDAI_API
#define UTF8_TO_TCHAR(x) x
#define TCHAR_TO_UTF8(x) x

constexpr int32 INDEX_NONE = -1;
constexpr float PI = 3.14159265358979323846f;
constexpr float KINDA_SMALL_NUMBER = 1.0e-4f;

template <typename T>
struct TNumericLimits : public std::numeric_limits<T>
{
    static constexpr T Max() { return std::numeric_limits<T>::max(); }
};

template <typename T>
constexpr std::remove_reference_t<T>&& MoveTemp(T&& value) noexcept
{
    return static_cast<std::remove_reference_t<T>&&>(value);
}

enum class EAllowShrinking
{
    No,
    Yes,
};

class FString;

class FName
{
public:
    FName() = default;
    FName(const char* value) : value_(value ? value : "") {}
    FName(const std::string& value) : value_(value) {}
    FName(const FString& value);

    bool IsNone() const { return value_.empty(); }
    const std::string& Str() const { return value_; }
    FString ToString() const;
    const char* c_str() const { return value_.c_str(); }

    friend bool operator==(const FName& lhs, const FName& rhs) { return lhs.value_ == rhs.value_; }
    friend bool operator!=(const FName& lhs, const FName& rhs) { return !(lhs == rhs); }
    friend bool operator<(const FName& lhs, const FName& rhs) { return lhs.value_ < rhs.value_; }

private:
    std::string value_;
};

inline const FName NAME_None{};

template <typename T>
class TArray
{
public:
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using reference = typename std::vector<T>::reference;
    using const_reference = typename std::vector<T>::const_reference;

    TArray() = default;
    TArray(std::initializer_list<T> init) : data_(init) {}

    int32 Num() const { return static_cast<int32>(data_.size()); }
    bool IsValidIndex(int32 index) const { return index >= 0 && index < Num(); }
    void Reset() { data_.clear(); }
    void Add(const T& value) { data_.push_back(value); }
    void Add(T&& value) { data_.push_back(std::move(value)); }
    void Init(const T& value, int32 count) { data_.assign(static_cast<size_t>(std::max(count, 0)), value); }
    void SetNum(int32 count) { data_.resize(static_cast<size_t>(std::max(count, 0))); }
    T& Last() { return data_.back(); }
    const T& Last() const { return data_.back(); }
    T* GetData() { return data_.data(); }
    const T* GetData() const { return data_.data(); }
    void RemoveAt(int32 index, int32 count = 1, EAllowShrinking = EAllowShrinking::Yes)
    {
        if (count <= 0 || !IsValidIndex(index)) return;
        const auto first = data_.begin() + index;
        const auto last = data_.begin() + std::min<int32>(Num(), index + count);
        data_.erase(first, last);
    }

    template <typename Pred>
    void Sort(Pred pred)
    {
        if constexpr (std::is_pointer_v<T>)
        {
            // Unreal's TArray::Sort dereferences pointer elements before
            // invoking the predicate.
            std::sort(data_.begin(), data_.end(), [&pred](T a, T b) {
                return pred(*a, *b);
            });
        }
        else
        {
            std::sort(data_.begin(), data_.end(), pred);
        }
    }

    reference operator[](int32 index) { return data_[static_cast<size_t>(index)]; }
    const_reference operator[](int32 index) const { return data_[static_cast<size_t>(index)]; }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    std::vector<T> data_;
};

template <typename K, typename V>
class TMap
{
public:
    bool Contains(const K& key) const { return data_.find(key) != data_.end(); }
    void Reset() { data_.clear(); }
    void Add(const K& key, const V& value) { data_[key] = value; }
    void Add(const K& key, V&& value) { data_[key] = std::move(value); }
    V& FindOrAdd(const K& key) { return data_[key]; }
    const V* Find(const K& key) const
    {
        auto it = data_.find(key);
        return it == data_.end() ? nullptr : &it->second;
    }
    V* Find(const K& key)
    {
        auto it = data_.find(key);
        return it == data_.end() ? nullptr : &it->second;
    }
    V FindRef(const K& key) const
    {
        auto it = data_.find(key);
        return it == data_.end() ? V{} : it->second;
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

private:
    std::map<K, V> data_;
};

template <typename K>
class TSet
{
public:
    TSet() = default;
    TSet(std::initializer_list<K> init) : data_(init) {}

    bool Contains(const K& key) const { return data_.find(key) != data_.end(); }

private:
    std::set<K> data_;
};

class FString
{
public:
    FString() = default;
    FString(const char* value) : value_(value ? value : "") {}
    FString(const std::string& value) : value_(value) {}

    static FString Chr(TCHAR ch)
    {
        return FString(std::string(1, ch));
    }

    bool IsEmpty() const { return value_.empty(); }
    int32 Len() const { return static_cast<int32>(value_.size()); }
    void Reset() { value_.clear(); }
    void AppendChar(TCHAR ch) { value_.push_back(ch); }
    bool StartsWith(const char* prefix) const
    {
        const std::string p = prefix ? prefix : "";
        return value_.rfind(p, 0) == 0;
    }
    bool EndsWith(const char* suffix) const
    {
        const std::string s = suffix ? suffix : "";
        return value_.size() >= s.size() && value_.compare(value_.size() - s.size(), s.size(), s) == 0;
    }
    bool Contains(const char* needle) const
    {
        return value_.find(needle ? needle : "") != std::string::npos;
    }
    bool FindChar(TCHAR ch, int32& outIndex) const
    {
        const auto pos = value_.find(ch);
        if (pos == std::string::npos) return false;
        outIndex = static_cast<int32>(pos);
        return true;
    }
    FString Left(int32 count) const
    {
        return FString(value_.substr(0, static_cast<size_t>(std::max(count, 0))));
    }
    FString LeftChop(int32 count) const
    {
        const int32 safe = std::max(0, Len() - std::max(count, 0));
        return Left(safe);
    }
    FString TrimStartAndEnd() const
    {
        FString out(*this);
        out.TrimStartAndEndInline();
        return out;
    }
    void TrimStartAndEndInline()
    {
        auto isSpace = [](unsigned char ch) { return std::isspace(ch) != 0; };
        auto beginIt = std::find_if_not(value_.begin(), value_.end(), isSpace);
        auto endIt = std::find_if_not(value_.rbegin(), value_.rend(), isSpace).base();
        value_ = beginIt < endIt ? std::string(beginIt, endIt) : std::string();
    }
    FString ToUpper() const
    {
        FString out(*this);
        std::transform(out.value_.begin(), out.value_.end(), out.value_.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
        return out;
    }
    void ReplaceInline(const char* from, const char* to)
    {
        const std::string src = from ? from : "";
        const std::string dst = to ? to : "";
        if (src.empty()) return;
        size_t pos = 0;
        while ((pos = value_.find(src, pos)) != std::string::npos)
        {
            value_.replace(pos, src.size(), dst);
            pos += dst.size();
        }
    }
    void ParseIntoArray(TArray<FString>& out, const char* delimiter, bool cullEmpty) const
    {
        out.Reset();
        const std::string delim = delimiter ? delimiter : "";
        if (delim.empty())
        {
            if (!cullEmpty || !value_.empty()) out.Add(*this);
            return;
        }
        size_t start = 0;
        while (start <= value_.size())
        {
            const size_t end = value_.find(delim, start);
            const std::string token = value_.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!cullEmpty || !token.empty()) out.Add(FString(token));
            if (end == std::string::npos) break;
            start = end + delim.size();
        }
    }
    void ParseIntoArrayLines(TArray<FString>& out, bool cullEmpty) const
    {
        out.Reset();
        size_t start = 0;
        while (start <= value_.size())
        {
            size_t end = value_.find('\n', start);
            std::string token = value_.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!token.empty() && token.back() == '\r') token.pop_back();
            if (!cullEmpty || !token.empty()) out.Add(FString(token));
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }

    const char* operator*() const { return value_.c_str(); }
    const char* c_str() const { return value_.c_str(); }
    const std::string& Str() const { return value_; }

    TCHAR operator[](int32 index) const { return value_[static_cast<size_t>(index)]; }

    auto begin() const { return value_.begin(); }
    auto end() const { return value_.end(); }

    friend bool operator==(const FString& lhs, const FString& rhs) { return lhs.value_ == rhs.value_; }
    friend bool operator!=(const FString& lhs, const FString& rhs) { return !(lhs == rhs); }
    friend bool operator==(const FString& lhs, const char* rhs) { return lhs.value_ == (rhs ? rhs : ""); }
    friend bool operator==(const char* lhs, const FString& rhs) { return rhs == lhs; }
    friend bool operator<(const FString& lhs, const FString& rhs) { return lhs.value_ < rhs.value_; }

private:
    std::string value_;
};

inline FName::FName(const FString& value) : value_(value.Str()) {}
inline FString FName::ToString() const { return FString(value_); }

class FText
{
public:
    static FText FromString(const FString& value) { return FText(value); }
    FString ToString() const { return value_; }

private:
    explicit FText(const FString& value) : value_(value) {}
    FString value_;
};

struct FChar
{
    static bool IsAlnum(TCHAR ch) { return std::isalnum(static_cast<unsigned char>(ch)) != 0; }
    static bool IsDigit(TCHAR ch) { return std::isdigit(static_cast<unsigned char>(ch)) != 0; }
    static bool IsWhitespace(TCHAR ch) { return std::isspace(static_cast<unsigned char>(ch)) != 0; }
    static TCHAR ToLower(TCHAR ch) { return static_cast<TCHAR>(std::tolower(static_cast<unsigned char>(ch))); }
};

struct FMath
{
    template <typename T>
    static T Clamp(T value, T minValue, T maxValue)
    {
        return std::clamp(value, minValue, maxValue);
    }

    template <typename T>
    static T Max(T a, T b)
    {
        return std::max(a, b);
    }

    template <typename T>
    static T Min(T a, T b)
    {
        return std::min(a, b);
    }

    template <typename T>
    static T Abs(T value)
    {
        using std::abs;
        return abs(value);
    }

    template <typename T, typename U>
    static auto Lerp(T a, T b, U alpha) -> decltype(a + (b - a) * alpha)
    {
        return a + (b - a) * alpha;
    }

    static int32 RoundToInt(float value) { return static_cast<int32>(std::lround(value)); }
    static float FloorToFloat(float value) { return std::floor(value); }
    static float Sqrt(float value) { return std::sqrt(value); }
    static double Sqrt(double value) { return std::sqrt(value); }
    static float Cos(float value) { return std::cos(value); }
    static float Exp(float value) { return std::exp(value); }
    static bool IsFinite(float value) { return std::isfinite(value); }
};


#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cwctype>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstdarg>

#define OFFGRIDAI_API
#define TEXT(x) x
using int32 = int32_t;
using int64 = int64_t;
using int16 = int16_t;
using uint8 = uint8_t;
using TCHAR = char;
static constexpr int32 INDEX_NONE = -1;
static constexpr float KINDA_SMALL_NUMBER = 1.e-4f;
enum class EAllowShrinking { No, Yes };

enum class ESearchCase { CaseSensitive, IgnoreCase };
enum class ESearchDir { FromStart, FromEnd };

struct FChar { static char ToLower(char c){ return (char)std::tolower((unsigned char)c); } static bool IsAlnum(char c){ return std::isalnum((unsigned char)c) != 0; } };

struct FMath {
    template<typename T> static T Clamp(T v, T lo, T hi){ return std::max(lo, std::min(hi, v)); }
    static float Clamp(float v, float lo, float hi){ return std::max(lo, std::min(hi, v)); }
    static double Clamp(double v, double lo, double hi){ return std::max(lo, std::min(hi, v)); }
    static int32 Clamp(int32 v, int32 lo, int32 hi){ return std::max(lo, std::min(hi, v)); }
    template<typename T> static T Max(T a,T b){ return std::max(a,b); }
    template<typename T> static T Min(T a,T b){ return std::min(a,b); }
    static float Abs(float v){ return std::fabs(v); }
    static double Abs(double v){ return std::fabs(v); }
    static int32 Abs(int32 v){ return std::abs(v); }
    static float Sqrt(float v){ return std::sqrt(v); }
    static int32 RoundToInt(float v){ return (int32)std::lround(v); }
    static float Lerp(float a,float b,float t){ return a + (b-a)*t; }
    template<typename T> static T Max3(T a,T b,T c){ return std::max(a,std::max(b,c)); }
    static bool IsFinite(float v){ return std::isfinite(v); }
    static bool IsNearlyEqual(float a,float b,float tol=1.e-4f){ return std::fabs(a-b)<=tol; }
};

class FString {
public:
    std::string S;
    FString() = default;
    FString(const char* c): S(c?c:"") {}
    FString(const std::string& s): S(s) {}
    FString(char c): S(1,c) {}
    int32 Len() const { return (int32)S.size(); }
    bool IsEmpty() const { return S.empty(); }
    char operator[](int32 i) const { return S[(size_t)i]; }
    char& operator[](int32 i) { return S[(size_t)i]; }
    const char* c_str() const { return S.c_str(); }
    const char* operator*() const { return S.c_str(); }
    void AppendChar(char c){ S.push_back(c); }
    void Reset(){ S.clear(); }
    void ReplaceInline(const char* from,const char* to){ *this = Replace(from,to); }
    std::string::iterator begin(){ return S.begin(); }
    std::string::iterator end(){ return S.end(); }
    std::string::const_iterator begin() const { return S.begin(); }
    std::string::const_iterator end() const { return S.end(); }
    std::string ToStdString() const { return S; }
    FString ToLower() const { std::string r=S; std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c){return (char)std::tolower(c);}); return FString(r); }
    FString Mid(int32 start, int32 count=INT32_MAX) const { if(start<0) start=0; if(start>Len()) return FString(); return FString(S.substr((size_t)start, count==INT32_MAX?std::string::npos:(size_t)count));}
    FString Left(int32 count) const { return FString(S.substr(0,(size_t)std::max(0,count))); }
    FString Right(int32 count) const { count=std::max(0,count); if(count>=Len()) return *this; return FString(S.substr(S.size()-count)); }
    void LeftChopInline(int32 count){ if(count<=0) return; if(count>=Len()) S.clear(); else S.erase(S.size()-count); }
    bool StartsWith(const char* p) const { std::string q=p?p:""; return S.rfind(q,0)==0; }
    bool StartsWith(const FString& p) const { return StartsWith(p.S.c_str()); }
    bool EndsWith(const char* p) const { std::string q=p?p:""; return S.size()>=q.size() && S.compare(S.size()-q.size(), q.size(), q)==0; }
    bool EndsWith(const FString& p) const { return EndsWith(p.S.c_str()); }
    bool Contains(const FString& sub) const { return S.find(sub.S)!=std::string::npos; }
    bool Contains(const char* sub) const { return S.find(sub?sub:"")!=std::string::npos; }
    bool Contains(const char* sub, ESearchCase cs) const {
        std::string a=S,b=sub?sub:"";
        if(cs==ESearchCase::IgnoreCase){ std::transform(a.begin(),a.end(),a.begin(),[](unsigned char c){return (char)std::tolower(c);}); std::transform(b.begin(),b.end(),b.begin(),[](unsigned char c){return (char)std::tolower(c);});}
        return a.find(b)!=std::string::npos;
    }
    bool FindChar(char c, int32& idx) const { auto p=S.find(c); if(p==std::string::npos){idx=INDEX_NONE; return false;} idx=(int32)p; return true; }
    int32 Find(const FString& sub, ESearchCase cs=ESearchCase::CaseSensitive, ESearchDir dir=ESearchDir::FromStart, int32 start=0) const {
        std::string a=S,b=sub.S;
        if(cs==ESearchCase::IgnoreCase){ std::transform(a.begin(),a.end(),a.begin(),[](unsigned char c){return (char)std::tolower(c);}); std::transform(b.begin(),b.end(),b.begin(),[](unsigned char c){return (char)std::tolower(c);});}
        auto pos=(dir==ESearchDir::FromStart)?a.find(b,(size_t)std::max(0,start)):a.rfind(b);
        return pos==std::string::npos?INDEX_NONE:(int32)pos;
    }
    FString Replace(const char* from, const char* to) const { std::string r=S; std::string f=from?from:"", t=to?to:""; if(f.empty()) return FString(r); size_t pos=0; while((pos=r.find(f,pos))!=std::string::npos){ r.replace(pos,f.size(),t); pos+=t.size(); } return FString(r); }
    static FString Chr(char c){ return FString(c); }
    static FString Printf(const char* fmt, ...){
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt ? fmt : "", args);
        va_end(args);
        buf[sizeof(buf)-1] = '\0';
        return FString(buf);
    }
    friend bool operator==(const FString&a,const FString&b){return a.S==b.S;}
    friend bool operator!=(const FString&a,const FString&b){return a.S!=b.S;}
    friend bool operator==(const FString&a,const char*b){return a.S==(b?b:"");}
    friend bool operator!=(const FString&a,const char*b){return !(a==b);}
    friend FString operator+(const FString&a,const FString&b){ return FString(a.S+b.S); }
};
namespace std { template<> struct hash<FString>{ size_t operator()(FString const& f) const noexcept { return hash<string>()(f.S); } }; }

class FName {
    std::string N;
public:
    FName() = default;
    FName(const char* c): N(c?c:"") {}
    FName(const FString& s): N(s.S) {}
    FString ToString() const { return FString(N); }
    bool IsNone() const { return N.empty(); }
    const std::string& ToStdString() const { return N; }
    friend bool operator==(const FName&a,const FName&b){return a.N==b.N;}
    friend bool operator!=(const FName&a,const FName&b){return a.N!=b.N;}
    friend bool operator==(const FName&a,const char*b){return a.N==(b?b:"");}
    friend bool operator!=(const FName&a,const char*b){return !(a==b);}
};
namespace std { template<> struct hash<FName>{ size_t operator()(FName const& f) const noexcept { return hash<string>()(f.ToStdString()); } }; }
static const FName NAME_None;

class FText {
    FString V;
public:
    FText() = default;
    FText(const FString& s): V(s) {}
    FText(const char* s): V(s) {}
    static FText FromString(const FString& s){ return FText(s); }
    FString ToString() const { return V; }
};

template<class T>
class TArray {
    std::vector<T> V;
public:
    using iterator=typename std::vector<T>::iterator; using const_iterator=typename std::vector<T>::const_iterator;
    TArray()=default; TArray(std::initializer_list<T> init): V(init) {}
    int32 Num() const { return (int32)V.size(); }
    void Add(const T& t){ V.push_back(t); }
    void Add(T&& t){ V.push_back(std::move(t)); }
    void Append(const TArray<T>& other){ for(const auto& x: other.Std()) V.push_back(x); }
    void AddUnique(const T& t){ if(std::find(V.begin(),V.end(),t)==V.end()) V.push_back(t); }
    T& AddDefaulted_GetRef(){ V.emplace_back(); return V.back(); }
    void Reset(){ V.clear(); }
    void Empty(){ V.clear(); }
    void Reserve(int32 n){ V.reserve(n); }
    void SetNumZeroed(int32 n){ V.assign((size_t)n, T{}); }
    void SetNum(int32 n){ V.resize((size_t)n); }
    bool IsValidIndex(int32 i) const { return i>=0 && i<Num(); }
    T& operator[](int32 i){ return V[(size_t)i]; }
    const T& operator[](int32 i) const { return V[(size_t)i]; }
    iterator begin(){ return V.begin(); } iterator end(){ return V.end(); }
    const_iterator begin() const { return V.begin(); } const_iterator end() const { return V.end(); }
    void Sort(){ std::sort(V.begin(), V.end()); }
    template<class Pred> void Sort(Pred p){ std::sort(V.begin(), V.end(), p); }
    auto rbegin(){return V.rbegin();} auto rend(){return V.rend();}
    T& Last(){ return V.back(); }
    const T& Last() const { return V.back(); }
    template<class Pred> int32 RemoveAll(Pred p){ auto old=V.size(); V.erase(std::remove_if(V.begin(), V.end(), p), V.end()); return (int32)(old-V.size()); }
    T* GetData(){ return V.data(); }
    const T* GetData() const { return V.data(); }
    void RemoveAt(int32 index, int32 count=1, EAllowShrinking=EAllowShrinking::Yes){ if(index<0||count<=0||index>=Num()) return; int32 end=std::min(index+count, Num()); V.erase(V.begin()+index, V.begin()+end); }
    int32 IndexOfByKey(const T& key) const { auto it=std::find(V.begin(),V.end(),key); return it==V.end()?INDEX_NONE:(int32)std::distance(V.begin(),it); }
    const std::vector<T>& Std() const { return V; }
};


template<class K,class V>
struct TPair { K Key; V Value; TPair(const K& k,const V& v):Key(k),Value(v){} };

template<class K,class V>
class TMap {
    std::unordered_map<K,V> M;
public:
    struct iterator {
        using base_it=typename std::unordered_map<K,V>::iterator; base_it It;
        iterator(base_it it):It(it){}
        iterator& operator++(){++It; return *this;}
        bool operator!=(const iterator& o) const {return It!=o.It;}
        TPair<K,V> operator*() const { return TPair<K,V>(It->first, It->second); }
    };
    struct const_iterator {
        using base_it=typename std::unordered_map<K,V>::const_iterator; base_it It;
        const_iterator(base_it it):It(it){}
        const_iterator& operator++(){++It; return *this;}
        bool operator!=(const const_iterator& o) const {return It!=o.It;}
        TPair<K,V> operator*() const { return TPair<K,V>(It->first, It->second); }
    };
    int32 Num() const { return (int32)M.size(); }
    void Reset(){ M.clear(); }
    V& Add(const K& k){ return M[k]; }
    void Add(const K& k,const V& v){ M[k]=v; }
    V& FindOrAdd(const K& k){ return M[k]; }
    V* Find(const K& k){ auto it=M.find(k); return it==M.end()?nullptr:&it->second; }
    const V* Find(const K& k) const { auto it=M.find(k); return it==M.end()?nullptr:&it->second; }
    bool Contains(const K& k) const { return M.find(k)!=M.end(); }
    V FindRef(const K& k) const { auto it=M.find(k); return it==M.end()?V{}:it->second; }
    V& FindChecked(const K& k){ return M.at(k); }
    const V& FindChecked(const K& k) const { return M.at(k); }
    void GetKeys(TArray<K>& Out) const { for(auto& kv:M) Out.Add(kv.first); }
    void GenerateValueArray(TArray<V>& Out) const { for(auto& kv:M) Out.Add(kv.second); }
    V& operator[](const K& k){ return M[k]; }
    iterator begin(){return iterator(M.begin());} iterator end(){return iterator(M.end());}
    const_iterator begin() const {return const_iterator(M.begin());} const_iterator end() const {return const_iterator(M.end());}
};

template<class T>
class TSet {
    std::unordered_set<T> S;
public:
    TSet()=default; TSet(std::initializer_list<T> init): S(init) {}
    int32 Num() const { return (int32)S.size(); }
    void Add(const T& t){ S.insert(t); }
    bool Contains(const T& t) const { return S.find(t)!=S.end(); }
    auto begin(){return S.begin();} auto end(){return S.end();}
    auto begin() const {return S.begin();} auto end() const {return S.end();}
};


template<typename T> struct TNumericLimits { static T Max(){ return std::numeric_limits<T>::max(); } static T Min(){ return std::numeric_limits<T>::lowest(); } };

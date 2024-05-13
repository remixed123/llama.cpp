#pragma once

/**
 * Provides a simple direct 1-level only config file logic
 * by Humans for All
 * 
 * ## File format
 * 
 * It can consist of multiple config groups.
 * * the group name needs to start at the begining of the line.
 * Each group can inturn contain multiple config fields (key:value pairs) wrt that group.
 * * the group fields need to have 1 or more space at the begining of line.
 * 
 * ## Supported data types
 * 
 * The fields can have values belonging to ane one of the below types
 * * strings - enclosed in double quotes
 *             this is also the fallback catch all type, but dont rely on this behaviour.
 * * int - using decimal number system
 * * float - needs to have a decimal point and or e/E
 *           if decimal point is used, there should be atleast one decimal number on its either side
 * * bool - either true or false
 * 
 * It tries to provide a crude expanded form of array wrt any of the above supported types.
 * For this one needs to define keys using the pattern TheKeyName-0, TheKeyName-1, ....
 * 
 * ## Additional notes
 * 
 * NativeCharSize encoded char refers to chars which fit within the size of char type in a given
 * type of c++ string or base bitsize of a encoding standard, like 1 byte in case of std::string,
 * utf-8, ...
 * * example english alphabets in utf-8 encoding space are 1byte chars, in its variable length
 *   encoding space.
 * 
 * MultiNativeCharSize encoded char refers to chars which occupy multiple base-char-bit-size of
 * a c++ string type or char encoding standard.
 * * example indian scripts alphabets in utf-8 encoding space occupy multiple bytes in its variable
 *   length encoding space.
 * 
 * Sane variable length encoding - refers to encoding where the values of NativeCharSized chars of
 * a char encoding space cant overlap with values in NativeCharSize subparts of MultiNativeCharSized
 * chars of the same char encoding standard.
 * * utf-8 shows this behaviour
 * * chances are utf-16 and utf-32 also show this behaviour (need to cross check once)
 */

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <regex>
#include <variant>
#include <sstream>
#include <cuchar>

#include "groupkv.hpp"


#undef SC_DEBUG_VERBOSE

#undef SC_STR_OVERSMART
#ifdef SC_STR_OVERSMART
#define str_trim str_trim_oversmart
#else
#define str_trim str_trim_dumb
#endif


// **** **** **** String related helpers **** **** **** //


inline size_t wcs_to_mbs(std::string &sDest, const std::wstring &wSrc) {
    std::mbstate_t mbState = std::mbstate_t();
    const wchar_t *wSrcP = wSrc.c_str();
    auto reqLen = std::wcsrtombs(nullptr, &wSrcP, 0, &mbState);
    sDest.resize(reqLen);
    return std::wcsrtombs(sDest.data(), &wSrcP, sDest.length(), &mbState);
}

inline size_t mbs_to_wcs(std::wstring &wDest, const std::string &sSrc) {
    std::mbstate_t mbState = std::mbstate_t();
    const char *sSrcP = sSrc.c_str();
    auto reqLen = std::mbsrtowcs(nullptr, &sSrcP, 0, &mbState);
    wDest.resize(reqLen);
    return std::mbsrtowcs(wDest.data(), &sSrcP, wDest.length(), &mbState);
}

template <typename TString>
inline void dumphex_string(const TString &sIn, const std::string &msgTag){
    LDBUG("%s[ ", msgTag.c_str());
    for(auto c: sIn) {
        auto cSize = sizeof(c);
        if (cSize == 1) {
            LDBUG("%02x, ", (uint8_t)c);
        } else if (cSize == 2) {
            LDBUG("%04x, ", (uint16_t)c);
        } else if (cSize == 4) {
            LDBUG("%08x, ", (uint32_t)c);
        } else {
            std::stringstream ss;
            ss << "ERRR:" << __func__ << ":Unsupported char type with size [" << cSize << "]";
            throw std::runtime_error( ss.str().c_str() );
        }
    }
    LDBUG_LN(" ]");
}

// Remove chars from begin and end of the passed string, provided the char
// belongs to one of the chars in trimChars.
//
// NOTE: This will work perfectly provided the string being trimmed as well as
// chars being trimmed are made up of NativeCharSize chars from same encoded space.
// For utf-8, this means the ascii equivalent 1byteSized chars of utf8 and not
// variable length MultiNativeCharSize (ie multibye in case of utf-8) ones.
// NOTE: It will also work, if atleast either end of string as well as trimChars
// have NativeCharSize chars from their encoding space, rather than variable
// length MultiNativeCharSize based chars if any. There needs to be NativeCharSized
// chars beyond any chars that get trimmed, on either side.
//
// NOTE: Given the way UTF-8 char encoding is designed, where NativeCharSize 1byte
// encoded chars are fully unique and dont overlap with any bytes from any of the
// variable length MultiNativeCharSize encoded chars in the utf-8 space, so as long as
// the trimChars belong to NativeCharSize chars subset, the logic should work, even
// if string has a mixture of NativeCharSize and MultiNativeCharSize encoded chars.
// Chances are utf-16 and utf-32 also have similar characteristics wrt thier
// NativeCharSize encoded chars (ie those fully encoded within single 16bit and 32bit 
// value respectively), and so equivalent semantic applies to them also.
//
// ALERT: Given that this simple minded logic, works at individual NativeCharSize level
// only, If trimChars involve variable length MultiNativeCharSize encoded chars, then
// * because different NativeCharSize subparts (bytes in case of utf-8) from different
//   MultiNativeCharSize trim chars when clubbed together can map to some other new char
//   in a variable length encoded char space, if there is that new char at either end
//   of the string, it may get trimmed, because of the possibility of mix up mentioned.
// * given that different variable length MultiNativeCharSize encoded chars may have
//   some common NativeCharSize subparts (bytes in case of utf-8) between them, if one
//   of these chars is at either end of the string and another char is in trimChars,
//   then string may get partially trimmed wrt such a char at either end.
//
template <typename TString>
inline TString str_trim_dumb(TString sin, const TString &trimChars=" \t\n") {
#ifdef SC_DEBUG_VERBOSE
    dumphex_string(sin, "DBUG:StrTrimDumb:Str:");
    dumphex_string(trimChars, "DBUG:StrTrimDumb:TrimChars:");
#endif
    sin.erase(sin.find_last_not_of(trimChars)+1);
    sin.erase(0, sin.find_first_not_of(trimChars));
    return sin;
}

// Remove chars from begin and end of the passed string, provided the char belongs
// to one of the chars in trimChars.
// NOTE: Internally converts to wchar/wstring to try and support proper trimming,
// wrt possibly more languages, to some extent. IE even if the passed string
// contains multibyte encoded characters in it in utf-8 space (ie MultiNativeCharSize),
// it may get converted to NativeCharSize chars in the expanded wchar_t encoding space,
// thus leading to fixed NativeCharSize driven logic itself handling things sufficiently.
// Look at str_trim_dumb comments for additional aspects.
inline std::string str_trim_oversmart(std::string sIn, const std::string &trimChars=" \t\n") {
    std::wstring wIn;
    mbs_to_wcs(wIn, sIn);
    std::wstring wTrimChars;
    mbs_to_wcs(wTrimChars, trimChars);
    auto wOut = str_trim_dumb(wIn, wTrimChars);
    std::string sOut;
    wcs_to_mbs(sOut, wOut);
    return sOut;
}

// Remove atmost 1 char at the begin and 1 char at the end of the passed string,
// provided the char belongs to one of the chars in trimChars.
//
// NOTE: Chars being trimmed (ie in trimChars) needs to be part of NativeCharSize
// subset of the string's encoded char space, to avoid mix up when working with
// strings which can be utf-8/utf-16/utf-32/sane-variable-length encoded strings.
//
// NOTE:UTF8: This will work provided the string being trimmed as well the chars
// being trimmed are made up of 1byte encoded chars in case of utf8 encoding space.
// If the string being trimmed includes multibyte (ie MultiNativeCharSize) encoded
// characters at either end, then trimming can mess things up, if you have multibyte
// encoded utf-8 chars in the trimChars set.
//
// Currently given that SimpCfg only uses this with NativeCharSize chars in the
// trimChars and most of the platforms are likely to be using utf-8 based char
// space (which is a realtively sane variable length char encoding from this
// logics perspective), so not providing oversmart variant.
//
template <typename TString>
inline TString str_trim_single(TString sin, const TString& trimChars=" \t\n") {
    if (sin.empty()) return sin;
    for(auto c: trimChars) {
        if (c == sin.front()) {
            sin = sin.substr(1, TString::npos);
            break;
        }
    }
    if (sin.empty()) return sin;
    for(auto c: trimChars) {
        if (c == sin.back()) {
            sin = sin.substr(0, sin.length()-1);
            break;
        }
    }
    return sin;
}

// Convert to lower case, if language has upper and lower case semantic
//
// This works for fixed size encoded char spaces.
//
// For variable length encoded char spaces, it can work
// * if one is doing the conversion for languages which fit into NativeCharSized chars in it
// * AND if one is working with a sane variable length encoding standard
// * ex: this will work if trying to do the conversion for english language within utf-8
//
template <typename TString>
inline TString str_tolower(const TString &sin) {
    TString sout;
    sout.resize(sin.size());
    std::transform(sin.begin(), sin.end(), sout.begin(), [](auto c)->auto {return std::tolower(c);});
#ifdef SC_DEBUG_VERBOSE
    dumphex_string(sin, "DBUG:StrToLower:in:");
    dumphex_string(sout, "DBUG:StrToLower:out:");
#endif
    return sout;
}

inline void str_compare_dump(const std::string &s1, const std::string &s2) {
    LDBUG_LN("DBUG:%s:%s:Len:%zu", __func__, s1.c_str(), s1.length());
    LDBUG_LN("DBUG:%s:%s:Len:%zu", __func__, s2.c_str(), s2.length());
    int minLen = s1.length() < s2.length() ? s1.length() : s2.length();
    for(int i=0; i<minLen; i++) {
        LDBUG_LN("DBUG:%s:%d:%c:%c", __func__, i, s1[i], s2[i]);
    }
}


template<typename TypeWithStrSupp>
std::string str(TypeWithStrSupp value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}


// **** **** **** the SimpCfg **** **** **** //


class SimpCfg : public GroupKV {

private:
    std::regex rInt {R"(^[-+]?\d+$)"};
    std::regex rFloat {R"(^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$)"};

public:

    SimpCfg(GroupKVMapMapVariant defaultMap) : GroupKV(defaultMap) {}


    void set_string(const std::string &group, const MultiPart &keyParts, const std::string &value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_bool(const std::string &group, const MultiPart &keyParts, bool value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_bool(const std::string &group, const MultiPart &keyParts, const std::string &value) {
        std::string sValue = str_tolower(value);
        bool bValue = sValue == "true" ? true : false;
        //LDBUG_LN("DBUG:%s:%s:%s:%d", __func__, value.c_str(), sValue.c_str(), bValue);
        set_bool(group, keyParts, bValue);
    }

    void set_int32(const std::string &group, const MultiPart &keyParts, int32_t value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_int32(const std::string &group, const MultiPart &keyParts, std::string &value) {
        auto ivalue = strtol(value.c_str(), nullptr, 0);
        set_int32(group, keyParts, ivalue);
    }

    void set_int64(const std::string &group, const MultiPart &keyParts, int64_t value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_int64(const std::string &group, const MultiPart &keyParts, std::string &value) {
        auto ivalue = strtoll(value.c_str(), nullptr, 0);
        set_int64(group, keyParts, ivalue);
    }

    void set_double(const std::string &group, const MultiPart &keyParts, double value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_double(const std::string &group, const MultiPart &keyParts, std::string &value) {
        auto dvalue = strtod(value.c_str(), nullptr);
        set_double(group, keyParts, dvalue);
    }


    std::string get_string(const std::string &group, const MultiPart &keyParts, const std::string &defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    bool get_bool(const std::string &group, const MultiPart &keyParts, bool defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    int32_t get_int32(const std::string &group, const MultiPart &keyParts, int32_t defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    int64_t get_int64(const std::string &group, const MultiPart &keyParts, int64_t defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    double get_double(const std::string &group, const MultiPart &keyParts, double defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }


    static void locale_prepare(std::string &sSavedLocale) {
        sSavedLocale = std::setlocale(LC_ALL, nullptr);
        auto sUpdatedLocale = std::setlocale(LC_ALL, "en_US.UTF-8");
        LDBUG_LN("DBUG:%s:Locale:Prev:%s:Cur:%s", __func__, sSavedLocale.c_str(), sUpdatedLocale);
    }

    static void locale_restore(const std::string &sSavedLocale) {
        auto sCurLocale = std::setlocale(LC_ALL, sSavedLocale.c_str());
        LDBUG_LN("DBUG:%s:Locale:Requested:%s:Got:%s", __func__, sSavedLocale.c_str(), sCurLocale);
    }

    void load(const std::string &fname) {
        std::ifstream f {fname};
        if (!f) {
            LERRR_LN("ERRR:SC:%s:%s:failed to load...", __func__, fname.c_str());
            throw std::runtime_error { "ERRR:SimpCfg:File not found" };
        } else {
            LDBUG_LN("DBUG:SC:%s:%s", __func__, fname.c_str());
        }
        std::string group;
        int iLine = 0;
        while(!f.eof()) {
            iLine += 1;
            std::string curL;
            getline(f, curL);
            if (curL.empty()) {
                continue;
            }
            if (curL[0] == '#') {
                continue;
            }
            bool bGroup = !isspace(curL[0]);
            curL = str_trim(curL);
            if (bGroup) {
                curL = str_trim_single(curL, {"\""});
                group = curL;
                LDBUG_LN("DBUG:SC:%s:group:%s", __func__, group.c_str());
                continue;
            }
            auto dPos = curL.find(':');
            if (dPos == std::string::npos) {
                LERRR_LN("ERRR:SC:%s:%d:invalid key value line:%s", __func__, iLine, curL.c_str());
                throw std::runtime_error { "ERRR:SimpCfg:Invalid key value line" };
            }
            auto dEnd = curL.length() - dPos;
            if ((dPos == 0) || (dEnd < 2)) {
                LERRR_LN("ERRR:SC:%s:%d:invalid key value line:%s", __func__, iLine, curL.c_str());
                throw std::runtime_error { "ERRR:SimpCfg:Invalid key value line" };
            }
            std::string key = curL.substr(0, dPos);
            key = str_trim(key);
            key = str_trim_single(key, {"\""});
            std::string value = curL.substr(dPos+1);
            value = str_trim(value);
            value = str_trim(value, {","});
            std::string vtype = "bool";
            auto valueLower = str_tolower(value);
            if ((valueLower.compare("true") == 0) || (valueLower == "false")) {
                set_bool(group, {key}, value);
            } else if (std::regex_match(value, rInt)) {
                vtype = "int";
                set_int64(group, {key}, value);
            } else if (std::regex_match(value, rFloat)) {
                vtype = "float";
                set_double(group, {key}, value);
            } else {
                vtype = "string";
                if (!value.empty() && (value.front() != '"')) {
                    LWARN_LN("WARN:SC:%s:%d:%s:k:%s:v:%s:is this string?", __func__, iLine, group.c_str(), key.c_str(), value.c_str());
                }
                value = str_trim_single(value, {"\""});
                set_string(group, {key}, value);
            }
            //LDBUG_LN("DBUG:SC:%s:%d:kv:%s:%s:%s:%s", __func__, iLine, group.c_str(), key.c_str(), vtype.c_str(), value.c_str());
        }
    }

};
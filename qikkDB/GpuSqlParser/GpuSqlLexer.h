
// Generated from C:/Users/AndrejFusekInstarea/Documents/GPU-DB/qikkDB/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, LSQR_BRC = 13, 
    RSQR_BRC = 14, STRING = 15, DELIMID = 16, DATELIT = 17, TIMELIT = 18, 
    POINT = 19, MULTIPOINT = 20, LINESTRING = 21, MULTILINESTRING = 22, 
    POLYGON = 23, MULTIPOLYGON = 24, INTTYPE = 25, LONGTYPE = 26, DATETYPE = 27, 
    DETETIMETYPE = 28, FLOATTYPE = 29, DOUBLETYPE = 30, STRINGTYPE = 31, 
    BOOLEANTYPE = 32, POINTTYPE = 33, POLYTYPE = 34, TYPES = 35, INDEX = 36, 
    UNIQUE = 37, PRIMARY = 38, KEY = 39, CREATE = 40, ADD = 41, DROP = 42, 
    ALTER = 43, RENAME = 44, SET = 45, INSERT = 46, INTO = 47, SHOW = 48, 
    SIZE = 49, FULL = 50, OUTER = 51, INNER = 52, DATABASE = 53, DATABASES = 54, 
    TABLE = 55, TABLES = 56, COLUMN = 57, COLUMNS = 58, BLOCK = 59, CONSTRAINTS = 60, 
    VALUES = 61, SELECT = 62, FROM = 63, JOIN = 64, WHERE = 65, GROUP = 66, 
    AS = 67, IN = 68, TO = 69, IS = 70, NOT = 71, NULL_T = 72, BY = 73, 
    BETWEEN = 74, ON = 75, ORDER = 76, DIR = 77, LIMIT = 78, OFFSET = 79, 
    QUERY = 80, AVG_AGG = 81, SUM_AGG = 82, MIN_AGG = 83, MAX_AGG = 84, 
    COUNT_AGG = 85, YEAR = 86, MONTH = 87, DAY = 88, HOUR = 89, MINUTE = 90, 
    SECOND = 91, WEEKDAY = 92, DAYOFWEEK = 93, NOW = 94, PI = 95, ABS = 96, 
    SIN = 97, COS = 98, TAN = 99, COT = 100, ASIN = 101, ACOS = 102, ATAN = 103, 
    ATAN2 = 104, LOG10 = 105, LOG = 106, EXP = 107, POW = 108, SQRT = 109, 
    SQUARE = 110, SIGN = 111, ROOT = 112, ROUND = 113, CEIL = 114, FLOOR = 115, 
    LTRIM = 116, RTRIM = 117, LOWER = 118, UPPER = 119, REVERSE = 120, LEN = 121, 
    LEFT = 122, RIGHT = 123, CONCAT = 124, CAST = 125, RETPAYLOAD = 126, 
    GEO_CONTAINS = 127, GEO_INTERSECT = 128, GEO_UNION = 129, GEO_LONGITUDE_TO_TILE_X = 130, 
    GEO_LATITUDE_TO_TILE_Y = 131, GEO_TILE_X_TO_LONGITUDE = 132, GEO_TILE_Y_TO_LATITUDE = 133, 
    PLUS = 134, MINUS = 135, ASTERISK = 136, DIVISION = 137, MODULO = 138, 
    XOR = 139, EQUALS = 140, NOTEQUALS = 141, NOTEQUALS_GT_LT = 142, LPAREN = 143, 
    RPAREN = 144, GREATER = 145, LESS = 146, GREATEREQ = 147, LESSEQ = 148, 
    LOGICAL_NOT = 149, OR = 150, AND = 151, BIT_OR = 152, BIT_AND = 153, 
    L_SHIFT = 154, R_SHIFT = 155, BOOLEANLIT = 156, TRUE = 157, FALSE = 158, 
    FLOATLIT = 159, INTLIT = 160, ID = 161
  };

  GpuSqlLexer(antlr4::CharStream *input);
  ~GpuSqlLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};


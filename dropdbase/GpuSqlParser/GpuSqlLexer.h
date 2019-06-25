
// Generated from C:/Users/mstano/GPU-DB/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, STRING = 13, 
    DATELIT = 14, TIMELIT = 15, POINT = 16, MULTIPOINT = 17, LINESTRING = 18, 
    MULTILINESTRING = 19, POLYGON = 20, MULTIPOLYGON = 21, DATATYPE = 22, 
    INTTYPE = 23, LONGTYPE = 24, FLOATTYPE = 25, DOUBLETYPE = 26, STRINGTYPE = 27, 
    BOOLEANTYPE = 28, POINTTYPE = 29, POLYTYPE = 30, INSERTINTO = 31, CREATEDB = 32, 
    DROPDB = 33, CREATETABLE = 34, DROPTABLE = 35, ALTERTABLE = 36, ADD = 37, 
    DROPCOLUMN = 38, ALTERCOLUMN = 39, CREATEINDEX = 40, INDEX = 41, PRIMARYKEY = 42, 
    VALUES = 43, SELECT = 44, FROM = 45, JOIN = 46, WHERE = 47, GROUPBY = 48, 
    AS = 49, IN = 50, ISNULL = 51, ISNOTNULL = 52, IS = 53, NULL = 54, NOT = 55, 
    BETWEEN = 56, ON = 57, ORDERBY = 58, DIR = 59, LIMIT = 60, OFFSET = 61, 
    SHOWDB = 62, SHOWTB = 63, SHOWCL = 64, AGG = 65, AVG = 66, SUM = 67, 
    MIN = 68, MAX = 69, COUNT = 70, YEAR = 71, MONTH = 72, DAY = 73, HOUR = 74, 
    MINUTE = 75, SECOND = 76, NOW = 77, PI = 78, ABS = 79, SIN = 80, COS = 81, 
    TAN = 82, COT = 83, ASIN = 84, ACOS = 85, ATAN = 86, ATAN2 = 87, LOG10 = 88, 
    LOG = 89, EXP = 90, POW = 91, SQRT = 92, SQUARE = 93, SIGN = 94, ROOT = 95, 
    ROUND = 96, CEIL = 97, FLOOR = 98, LTRIM = 99, RTRIM = 100, LOWER = 101, 
    UPPER = 102, REVERSE = 103, LEN = 104, LEFT = 105, RIGHT = 106, CONCAT = 107, 
    GEO_CONTAINS = 108, GEO_INTERSECT = 109, GEO_UNION = 110, PLUS = 111, 
    MINUS = 112, ASTERISK = 113, DIVISION = 114, MODULO = 115, XOR = 116, 
    EQUALS = 117, NOTEQUALS = 118, NOTEQUALS_GT_LT = 119, LPAREN = 120, 
    RPAREN = 121, GREATER = 122, LESS = 123, GREATEREQ = 124, LESSEQ = 125, 
    LOGICAL_NOT = 126, OR = 127, AND = 128, BIT_OR = 129, BIT_AND = 130, 
    L_SHIFT = 131, R_SHIFT = 132, BOOLEANLIT = 133, TRUE = 134, FALSE = 135, 
    FLOATLIT = 136, INTLIT = 137, ID = 138
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


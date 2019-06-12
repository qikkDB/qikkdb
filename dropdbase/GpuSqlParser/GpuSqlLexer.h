
// Generated from C:/Users/mstano/dropdbase_instarea/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, STRING = 13, 
    DATELIT = 14, TIMELIT = 15, DATATYPE = 16, POINT = 17, MULTIPOINT = 18, 
    LINESTRING = 19, MULTILINESTRING = 20, POLYGON = 21, MULTIPOLYGON = 22, 
    INTTYPE = 23, LONGTYPE = 24, FLOATTYPE = 25, DOUBLETYPE = 26, STRINGTYPE = 27, 
    BOOLEANTYPE = 28, POINTTYPE = 29, POLYTYPE = 30, INSERTINTO = 31, CREATEDB = 32, 
    CREATETABLE = 33, VALUES = 34, SELECT = 35, FROM = 36, JOIN = 37, WHERE = 38, 
    GROUPBY = 39, AS = 40, IN = 41, BETWEEN = 42, ON = 43, ORDERBY = 44, 
    DIR = 45, LIMIT = 46, OFFSET = 47, SHOWDB = 48, SHOWTB = 49, SHOWCL = 50, 
    AGG = 51, AVG = 52, SUM = 53, MIN = 54, MAX = 55, COUNT = 56, YEAR = 57, 
    MONTH = 58, DAY = 59, HOUR = 60, MINUTE = 61, SECOND = 62, NOW = 63, 
    PI = 64, ABS = 65, SIN = 66, COS = 67, TAN = 68, COT = 69, ASIN = 70, 
    ACOS = 71, ATAN = 72, ATAN2 = 73, LOG10 = 74, LOG = 75, EXP = 76, POW = 77, 
    SQRT = 78, SQUARE = 79, SIGN = 80, ROOT = 81, ROUND = 82, CEIL = 83, 
    FLOOR = 84, LTRIM = 85, RTRIM = 86, LOWER = 87, UPPER = 88, REVERSE = 89, 
    LEN = 90, LEFT = 91, RIGHT = 92, CONCAT = 93, GEO_CONTAINS = 94, GEO_INTERSECT = 95, 
    GEO_UNION = 96, PLUS = 97, MINUS = 98, ASTERISK = 99, DIVISION = 100, 
    MODULO = 101, XOR = 102, EQUALS = 103, NOTEQUALS = 104, NOTEQUALS_GT_LT = 105, 
    LPAREN = 106, RPAREN = 107, GREATER = 108, LESS = 109, GREATEREQ = 110, 
    LESSEQ = 111, NOT = 112, OR = 113, AND = 114, BIT_OR = 115, BIT_AND = 116, 
    L_SHIFT = 117, R_SHIFT = 118, FLOATLIT = 119, INTLIT = 120, ID = 121, 
    BOOLEANLIT = 122
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


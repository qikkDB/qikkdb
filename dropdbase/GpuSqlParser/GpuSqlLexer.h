
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
    FLOOR = 84, LTRIM = 85, RTRIM = 86, LOWER = 87, UPPER = 88, LEN = 89, 
    CONCAT = 90, GEO_CONTAINS = 91, GEO_INTERSECT = 92, GEO_UNION = 93, 
    PLUS = 94, MINUS = 95, ASTERISK = 96, DIVISION = 97, MODULO = 98, XOR = 99, 
    EQUALS = 100, NOTEQUALS = 101, NOTEQUALS_GT_LT = 102, LPAREN = 103, 
    RPAREN = 104, GREATER = 105, LESS = 106, GREATEREQ = 107, LESSEQ = 108, 
    NOT = 109, OR = 110, AND = 111, BIT_OR = 112, BIT_AND = 113, L_SHIFT = 114, 
    R_SHIFT = 115, FLOATLIT = 116, INTLIT = 117, ID = 118, BOOLEANLIT = 119
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


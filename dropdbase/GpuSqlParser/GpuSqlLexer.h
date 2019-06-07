
// Generated from C:/Users/mstano/dropdbase_instarea/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, DATELIT = 13, 
    TIMELIT = 14, DATATYPE = 15, POINT = 16, MULTIPOINT = 17, LINESTRING = 18, 
    MULTILINESTRING = 19, POLYGON = 20, MULTIPOLYGON = 21, INTTYPE = 22, 
    LONGTYPE = 23, FLOATTYPE = 24, DOUBLETYPE = 25, STRINGTYPE = 26, BOOLEANTYPE = 27, 
    POINTTYPE = 28, POLYTYPE = 29, INSERTINTO = 30, CREATEDB = 31, CREATETABLE = 32, 
    CREATEINDEX = 33, INDEX = 34, PRIMARYKEY = 35, VALUES = 36, SELECT = 37, 
    FROM = 38, JOIN = 39, WHERE = 40, GROUPBY = 41, AS = 42, IN = 43, BETWEEN = 44, 
    ON = 45, ORDERBY = 46, DIR = 47, LIMIT = 48, OFFSET = 49, SHOWDB = 50, 
    SHOWTB = 51, SHOWCL = 52, AGG = 53, AVG = 54, SUM = 55, MIN = 56, MAX = 57, 
    COUNT = 58, YEAR = 59, MONTH = 60, DAY = 61, HOUR = 62, MINUTE = 63, 
    SECOND = 64, NOW = 65, PI = 66, ABS = 67, SIN = 68, COS = 69, TAN = 70, 
    COT = 71, ASIN = 72, ACOS = 73, ATAN = 74, ATAN2 = 75, LOG10 = 76, LOG = 77, 
    EXP = 78, POW = 79, SQRT = 80, SQUARE = 81, SIGN = 82, ROOT = 83, ROUND = 84, 
    CEIL = 85, FLOOR = 86, GEO_CONTAINS = 87, GEO_INTERSECT = 88, GEO_UNION = 89, 
    PLUS = 90, MINUS = 91, ASTERISK = 92, DIVISION = 93, MODULO = 94, XOR = 95, 
    EQUALS = 96, NOTEQUALS = 97, NOTEQUALS_GT_LT = 98, LPAREN = 99, RPAREN = 100, 
    GREATER = 101, LESS = 102, GREATEREQ = 103, LESSEQ = 104, NOT = 105, 
    OR = 106, AND = 107, BIT_OR = 108, BIT_AND = 109, L_SHIFT = 110, R_SHIFT = 111, 
    FLOATLIT = 112, INTLIT = 113, ID = 114, BOOLEANLIT = 115, STRINGLIT = 116
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


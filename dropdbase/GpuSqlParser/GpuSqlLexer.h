
// Generated from C:/Users/David/source/repos/dropdbase_instarea/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

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
    VALUES = 33, SELECT = 34, FROM = 35, JOIN = 36, WHERE = 37, GROUPBY = 38, 
    AS = 39, IN = 40, BETWEEN = 41, ON = 42, ORDERBY = 43, DIR = 44, LIMIT = 45, 
    OFFSET = 46, SHOWDB = 47, SHOWTB = 48, SHOWCL = 49, AGG = 50, AVG = 51, 
    SUM = 52, MIN = 53, MAX = 54, COUNT = 55, YEAR = 56, MONTH = 57, DAY = 58, 
    HOUR = 59, MINUTE = 60, SECOND = 61, NOW = 62, PI = 63, ABS = 64, SIN = 65, 
    COS = 66, TAN = 67, ASIN = 68, ACOS = 69, ATAN = 70, LOG10 = 71, LOG = 72, 
    EXP = 73, POW = 74, SQRT = 75, SQUARE = 76, SIGN = 77, ROOT = 78, GEO_CONTAINS = 79, 
    GEO_INTERSECT = 80, GEO_UNION = 81, PLUS = 82, MINUS = 83, ASTERISK = 84, 
    DIVISION = 85, MODULO = 86, XOR = 87, EQUALS = 88, NOTEQUALS = 89, NOTEQUALS_GT_LT = 90, 
    LPAREN = 91, RPAREN = 92, GREATER = 93, LESS = 94, GREATEREQ = 95, LESSEQ = 96, 
    NOT = 97, OR = 98, AND = 99, BIT_OR = 100, BIT_AND = 101, L_SHIFT = 102, 
    R_SHIFT = 103, FLOATLIT = 104, INTLIT = 105, ID = 106, BOOLEANLIT = 107, 
    STRINGLIT = 108
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


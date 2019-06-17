
// Generated from C:/Users/mstano/GPU-DB/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, DATELIT = 13, 
    TIMELIT = 14, POINT = 15, MULTIPOINT = 16, LINESTRING = 17, MULTILINESTRING = 18, 
    POLYGON = 19, MULTIPOLYGON = 20, DATATYPE = 21, INTTYPE = 22, LONGTYPE = 23, 
    FLOATTYPE = 24, DOUBLETYPE = 25, STRINGTYPE = 26, BOOLEANTYPE = 27, 
    POINTTYPE = 28, POLYTYPE = 29, INSERTINTO = 30, CREATEDB = 31, DROPDB = 32, 
    CREATETABLE = 33, DROPTABLE = 34, ALTERTABLE = 35, ADD = 36, DROPCOLUMN = 37, 
    ALTERCOLUMN = 38, CREATEINDEX = 39, INDEX = 40, PRIMARYKEY = 41, VALUES = 42, 
    SELECT = 43, FROM = 44, JOIN = 45, WHERE = 46, GROUPBY = 47, AS = 48, 
    IN = 49, BETWEEN = 50, ON = 51, ORDERBY = 52, DIR = 53, LIMIT = 54, 
    OFFSET = 55, SHOWDB = 56, SHOWTB = 57, SHOWCL = 58, AGG = 59, AVG = 60, 
    SUM = 61, MIN = 62, MAX = 63, COUNT = 64, YEAR = 65, MONTH = 66, DAY = 67, 
    HOUR = 68, MINUTE = 69, SECOND = 70, NOW = 71, PI = 72, ABS = 73, SIN = 74, 
    COS = 75, TAN = 76, COT = 77, ASIN = 78, ACOS = 79, ATAN = 80, ATAN2 = 81, 
    LOG10 = 82, LOG = 83, EXP = 84, POW = 85, SQRT = 86, SQUARE = 87, SIGN = 88, 
    ROOT = 89, ROUND = 90, CEIL = 91, FLOOR = 92, GEO_CONTAINS = 93, GEO_INTERSECT = 94, 
    GEO_UNION = 95, PLUS = 96, MINUS = 97, ASTERISK = 98, DIVISION = 99, 
    MODULO = 100, XOR = 101, EQUALS = 102, NOTEQUALS = 103, NOTEQUALS_GT_LT = 104, 
    LPAREN = 105, RPAREN = 106, GREATER = 107, LESS = 108, GREATEREQ = 109, 
    LESSEQ = 110, NOT = 111, OR = 112, AND = 113, BIT_OR = 114, BIT_AND = 115, 
    L_SHIFT = 116, R_SHIFT = 117, BOOLEANLIT = 118, TRUE = 119, FALSE = 120, 
    FLOATLIT = 121, INTLIT = 122, ID = 123, STRINGLIT = 124
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


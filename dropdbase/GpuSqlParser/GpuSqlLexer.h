
// Generated from C:/Users/mstano/dropdbase_instarea/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

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
    AS = 49, IN = 50, BETWEEN = 51, ON = 52, ORDERBY = 53, DIR = 54, LIMIT = 55, 
    OFFSET = 56, SHOWDB = 57, SHOWTB = 58, SHOWCL = 59, AGG = 60, AVG = 61, 
    SUM = 62, MIN = 63, MAX = 64, COUNT = 65, YEAR = 66, MONTH = 67, DAY = 68, 
    HOUR = 69, MINUTE = 70, SECOND = 71, NOW = 72, PI = 73, ABS = 74, SIN = 75, 
    COS = 76, TAN = 77, COT = 78, ASIN = 79, ACOS = 80, ATAN = 81, ATAN2 = 82, 
    LOG10 = 83, LOG = 84, EXP = 85, POW = 86, SQRT = 87, SQUARE = 88, SIGN = 89, 
    ROOT = 90, ROUND = 91, CEIL = 92, FLOOR = 93, LTRIM = 94, RTRIM = 95, 
    LOWER = 96, UPPER = 97, REVERSE = 98, LEN = 99, LEFT = 100, RIGHT = 101, 
    CONCAT = 102, GEO_CONTAINS = 103, GEO_INTERSECT = 104, GEO_UNION = 105, 
    PLUS = 106, MINUS = 107, ASTERISK = 108, DIVISION = 109, MODULO = 110, 
    XOR = 111, EQUALS = 112, NOTEQUALS = 113, NOTEQUALS_GT_LT = 114, LPAREN = 115, 
    RPAREN = 116, GREATER = 117, LESS = 118, GREATEREQ = 119, LESSEQ = 120, 
    NOT = 121, OR = 122, AND = 123, BIT_OR = 124, BIT_AND = 125, L_SHIFT = 126, 
    R_SHIFT = 127, BOOLEANLIT = 128, TRUE = 129, FALSE = 130, FLOATLIT = 131, 
    INTLIT = 132, ID = 133
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


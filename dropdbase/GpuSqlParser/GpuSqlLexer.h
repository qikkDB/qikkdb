
// Generated from C:/Users/mstano/GPU-DB/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, STRING = 13, 
    DELIMID = 14, DATELIT = 15, TIMELIT = 16, POINT = 17, MULTIPOINT = 18, 
    LINESTRING = 19, MULTILINESTRING = 20, POLYGON = 21, MULTIPOLYGON = 22, 
    DATATYPE = 23, INTTYPE = 24, LONGTYPE = 25, FLOATTYPE = 26, DOUBLETYPE = 27, 
    STRINGTYPE = 28, BOOLEANTYPE = 29, POINTTYPE = 30, POLYTYPE = 31, INSERTINTO = 32, 
    CREATEDB = 33, DROPDB = 34, CREATETABLE = 35, DROPTABLE = 36, ALTERTABLE = 37, 
    ADD = 38, DROPCOLUMN = 39, ALTERCOLUMN = 40, CREATEINDEX = 41, INDEX = 42, 
    PRIMARYKEY = 43, VALUES = 44, SELECT = 45, FROM = 46, JOIN = 47, WHERE = 48, 
    GROUPBY = 49, AS = 50, IN = 51, BETWEEN = 52, ON = 53, ORDERBY = 54, 
    DIR = 55, LIMIT = 56, OFFSET = 57, SHOWDB = 58, SHOWTB = 59, SHOWCL = 60, 
    AGG = 61, AVG = 62, SUM = 63, MIN = 64, MAX = 65, COUNT = 66, YEAR = 67, 
    MONTH = 68, DAY = 69, HOUR = 70, MINUTE = 71, SECOND = 72, NOW = 73, 
    PI = 74, ABS = 75, SIN = 76, COS = 77, TAN = 78, COT = 79, ASIN = 80, 
    ACOS = 81, ATAN = 82, ATAN2 = 83, LOG10 = 84, LOG = 85, EXP = 86, POW = 87, 
    SQRT = 88, SQUARE = 89, SIGN = 90, ROOT = 91, ROUND = 92, CEIL = 93, 
    FLOOR = 94, LTRIM = 95, RTRIM = 96, LOWER = 97, UPPER = 98, REVERSE = 99, 
    LEN = 100, LEFT = 101, RIGHT = 102, CONCAT = 103, GEO_CONTAINS = 104, 
    GEO_INTERSECT = 105, GEO_UNION = 106, PLUS = 107, MINUS = 108, ASTERISK = 109, 
    DIVISION = 110, MODULO = 111, XOR = 112, EQUALS = 113, NOTEQUALS = 114, 
    NOTEQUALS_GT_LT = 115, LPAREN = 116, RPAREN = 117, GREATER = 118, LESS = 119, 
    GREATEREQ = 120, LESSEQ = 121, NOT = 122, OR = 123, AND = 124, BIT_OR = 125, 
    BIT_AND = 126, L_SHIFT = 127, R_SHIFT = 128, BOOLEANLIT = 129, TRUE = 130, 
    FALSE = 131, FLOATLIT = 132, INTLIT = 133, ID = 134
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


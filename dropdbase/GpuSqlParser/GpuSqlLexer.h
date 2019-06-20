
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
    AS = 49, IN = 50, BETWEEN = 51, ON = 52, ORDERBY = 53, DIR = 54, LIMIT = 55, 
    OFFSET = 56, INNER = 57, FULLOUTER = 58, SHOWDB = 59, SHOWTB = 60, SHOWCL = 61, 
    AGG = 62, AVG = 63, SUM = 64, MIN = 65, MAX = 66, COUNT = 67, YEAR = 68, 
    MONTH = 69, DAY = 70, HOUR = 71, MINUTE = 72, SECOND = 73, NOW = 74, 
    PI = 75, ABS = 76, SIN = 77, COS = 78, TAN = 79, COT = 80, ASIN = 81, 
    ACOS = 82, ATAN = 83, ATAN2 = 84, LOG10 = 85, LOG = 86, EXP = 87, POW = 88, 
    SQRT = 89, SQUARE = 90, SIGN = 91, ROOT = 92, ROUND = 93, CEIL = 94, 
    FLOOR = 95, LTRIM = 96, RTRIM = 97, LOWER = 98, UPPER = 99, REVERSE = 100, 
    LEN = 101, LEFT = 102, RIGHT = 103, CONCAT = 104, GEO_CONTAINS = 105, 
    GEO_INTERSECT = 106, GEO_UNION = 107, PLUS = 108, MINUS = 109, ASTERISK = 110, 
    DIVISION = 111, MODULO = 112, XOR = 113, EQUALS = 114, NOTEQUALS = 115, 
    NOTEQUALS_GT_LT = 116, LPAREN = 117, RPAREN = 118, GREATER = 119, LESS = 120, 
    GREATEREQ = 121, LESSEQ = 122, NOT = 123, OR = 124, AND = 125, BIT_OR = 126, 
    BIT_AND = 127, L_SHIFT = 128, R_SHIFT = 129, BOOLEANLIT = 130, TRUE = 131, 
    FALSE = 132, FLOATLIT = 133, INTLIT = 134, ID = 135
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



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
    DIR = 55, LIMIT = 56, OFFSET = 57, INNER = 58, FULLOUTER = 59, SHOWDB = 60, 
    SHOWTB = 61, SHOWCL = 62, AGG = 63, AVG = 64, SUM = 65, MIN = 66, MAX = 67, 
    COUNT = 68, YEAR = 69, MONTH = 70, DAY = 71, HOUR = 72, MINUTE = 73, 
    SECOND = 74, NOW = 75, PI = 76, ABS = 77, SIN = 78, COS = 79, TAN = 80, 
    COT = 81, ASIN = 82, ACOS = 83, ATAN = 84, ATAN2 = 85, LOG10 = 86, LOG = 87, 
    EXP = 88, POW = 89, SQRT = 90, SQUARE = 91, SIGN = 92, ROOT = 93, ROUND = 94, 
    CEIL = 95, FLOOR = 96, LTRIM = 97, RTRIM = 98, LOWER = 99, UPPER = 100, 
    REVERSE = 101, LEN = 102, LEFT = 103, RIGHT = 104, CONCAT = 105, GEO_CONTAINS = 106, 
    GEO_INTERSECT = 107, GEO_UNION = 108, PLUS = 109, MINUS = 110, ASTERISK = 111, 
    DIVISION = 112, MODULO = 113, XOR = 114, EQUALS = 115, NOTEQUALS = 116, 
    NOTEQUALS_GT_LT = 117, LPAREN = 118, RPAREN = 119, GREATER = 120, LESS = 121, 
    GREATEREQ = 122, LESSEQ = 123, NOT = 124, OR = 125, AND = 126, BIT_OR = 127, 
    BIT_AND = 128, L_SHIFT = 129, R_SHIFT = 130, BOOLEANLIT = 131, TRUE = 132, 
    FALSE = 133, FLOATLIT = 134, INTLIT = 135, ID = 136
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


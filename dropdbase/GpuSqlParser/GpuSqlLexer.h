
// Generated from C:/GPU-DB/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, LSQR_BRC = 13, 
    RSQR_BRC = 14, STRING = 15, DELIMID = 16, DATELIT = 17, TIMELIT = 18, 
    POINT = 19, MULTIPOINT = 20, LINESTRING = 21, MULTILINESTRING = 22, 
    POLYGON = 23, MULTIPOLYGON = 24, DATATYPE = 25, INTTYPE = 26, LONGTYPE = 27, 
    DATETYPE = 28, FLOATTYPE = 29, DOUBLETYPE = 30, STRINGTYPE = 31, BOOLEANTYPE = 32, 
    POINTTYPE = 33, POLYTYPE = 34, INSERTINTO = 35, CREATEDB = 36, DROPDB = 37, 
    CREATETABLE = 38, DROPTABLE = 39, ALTERTABLE = 40, ADD = 41, DROPCOLUMN = 42, 
    ALTERCOLUMN = 43, CREATEINDEX = 44, INDEX = 45, PRIMARYKEY = 46, VALUES = 47, 
    SELECT = 48, FROM = 49, JOIN = 50, WHERE = 51, GROUPBY = 52, AS = 53, 
    IN = 54, ISNULL = 55, ISNOTNULL = 56, BETWEEN = 57, ON = 58, ORDERBY = 59, 
    DIR = 60, LIMIT = 61, OFFSET = 62, INNER = 63, FULLOUTER = 64, SHOWDB = 65, 
    SHOWTB = 66, SHOWCL = 67, AVG_AGG = 68, SUM_AGG = 69, MIN_AGG = 70, 
    MAX_AGG = 71, COUNT_AGG = 72, YEAR = 73, MONTH = 74, DAY = 75, HOUR = 76, 
    MINUTE = 77, SECOND = 78, NOW = 79, PI = 80, ABS = 81, SIN = 82, COS = 83, 
    TAN = 84, COT = 85, ASIN = 86, ACOS = 87, ATAN = 88, ATAN2 = 89, LOG10 = 90, 
    LOG = 91, EXP = 92, POW = 93, SQRT = 94, SQUARE = 95, SIGN = 96, ROOT = 97, 
    ROUND = 98, CEIL = 99, FLOOR = 100, LTRIM = 101, RTRIM = 102, LOWER = 103, 
    UPPER = 104, REVERSE = 105, LEN = 106, LEFT = 107, RIGHT = 108, CONCAT = 109, 
    CAST = 110, GEO_CONTAINS = 111, GEO_INTERSECT = 112, GEO_UNION = 113, 
    PLUS = 114, MINUS = 115, ASTERISK = 116, DIVISION = 117, MODULO = 118, 
    XOR = 119, EQUALS = 120, NOTEQUALS = 121, NOTEQUALS_GT_LT = 122, LPAREN = 123, 
    RPAREN = 124, GREATER = 125, LESS = 126, GREATEREQ = 127, LESSEQ = 128, 
    LOGICAL_NOT = 129, OR = 130, AND = 131, BIT_OR = 132, BIT_AND = 133, 
    L_SHIFT = 134, R_SHIFT = 135, BOOLEANLIT = 136, TRUE = 137, FALSE = 138, 
    FLOATLIT = 139, INTLIT = 140, NULLLIT = 141, ID = 142
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


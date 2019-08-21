
// Generated from C:/GPU-DB/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, STRING = 13, 
    DELIMID = 14, DATELIT = 15, TIMELIT = 16, POINT = 17, MULTIPOINT = 18, 
    LINESTRING = 19, MULTILINESTRING = 20, POLYGON = 21, MULTIPOLYGON = 22, 
    DATATYPE = 23, INTTYPE = 24, LONGTYPE = 25, DATETYPE = 26, FLOATTYPE = 27, 
    DOUBLETYPE = 28, STRINGTYPE = 29, BOOLEANTYPE = 30, POINTTYPE = 31, 
    POLYTYPE = 32, INSERTINTO = 33, CREATEDB = 34, DROPDB = 35, CREATETABLE = 36, 
    DROPTABLE = 37, ALTERTABLE = 38, ADD = 39, DROPCOLUMN = 40, ALTERCOLUMN = 41, 
    CREATEINDEX = 42, INDEX = 43, PRIMARYKEY = 44, VALUES = 45, SELECT = 46, 
    FROM = 47, JOIN = 48, WHERE = 49, GROUPBY = 50, AS = 51, IN = 52, ISNULL = 53, 
    ISNOTNULL = 54, BETWEEN = 55, ON = 56, ORDERBY = 57, DIR = 58, LIMIT = 59, 
    OFFSET = 60, INNER = 61, FULLOUTER = 62, SHOWDB = 63, SHOWTB = 64, SHOWCL = 65, 
    AVG_AGG = 66, SUM_AGG = 67, MIN_AGG = 68, MAX_AGG = 69, COUNT_AGG = 70, 
    YEAR = 71, MONTH = 72, DAY = 73, HOUR = 74, MINUTE = 75, SECOND = 76, 
    NOW = 77, PI = 78, ABS = 79, SIN = 80, COS = 81, TAN = 82, COT = 83, 
    ASIN = 84, ACOS = 85, ATAN = 86, ATAN2 = 87, LOG10 = 88, LOG = 89, EXP = 90, 
    POW = 91, SQRT = 92, SQUARE = 93, SIGN = 94, ROOT = 95, ROUND = 96, 
    CEIL = 97, FLOOR = 98, LTRIM = 99, RTRIM = 100, LOWER = 101, UPPER = 102, 
    REVERSE = 103, LEN = 104, LEFT = 105, RIGHT = 106, CONCAT = 107, CAST = 108, 
    GEO_CONTAINS = 109, GEO_INTERSECT = 110, GEO_UNION = 111, PLUS = 112, 
    MINUS = 113, ASTERISK = 114, DIVISION = 115, MODULO = 116, XOR = 117, 
    EQUALS = 118, NOTEQUALS = 119, NOTEQUALS_GT_LT = 120, LPAREN = 121, 
    RPAREN = 122, GREATER = 123, LESS = 124, GREATEREQ = 125, LESSEQ = 126, 
    LOGICAL_NOT = 127, OR = 128, AND = 129, BIT_OR = 130, BIT_AND = 131, 
    L_SHIFT = 132, R_SHIFT = 133, BOOLEANLIT = 134, TRUE = 135, FALSE = 136, 
    FLOATLIT = 137, INTLIT = 138, NULLLIT = 139, ID = 140
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


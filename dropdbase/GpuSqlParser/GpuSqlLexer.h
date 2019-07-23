
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
    DATATYPE = 23, INTTYPE = 24, LONGTYPE = 25, DATETYPE = 26, FLOATTYPE = 27, 
    DOUBLETYPE = 28, STRINGTYPE = 29, BOOLEANTYPE = 30, POINTTYPE = 31, 
    POLYTYPE = 32, INSERTINTO = 33, CREATEDB = 34, DROPDB = 35, CREATETABLE = 36, 
    DROPTABLE = 37, ALTERTABLE = 38, ADD = 39, DROPCOLUMN = 40, ALTERCOLUMN = 41, 
    CREATEINDEX = 42, INDEX = 43, PRIMARYKEY = 44, VALUES = 45, SELECT = 46, 
    FROM = 47, JOIN = 48, WHERE = 49, GROUPBY = 50, AS = 51, IN = 52, BETWEEN = 53, 
    ON = 54, ORDERBY = 55, DIR = 56, LIMIT = 57, OFFSET = 58, INNER = 59, 
    FULLOUTER = 60, SHOWDB = 61, SHOWTB = 62, SHOWCL = 63, AGG = 64, AVG = 65, 
    SUM = 66, MIN = 67, MAX = 68, COUNT = 69, YEAR = 70, MONTH = 71, DAY = 72, 
    HOUR = 73, MINUTE = 74, SECOND = 75, NOW = 76, PI = 77, ABS = 78, SIN = 79, 
    COS = 80, TAN = 81, COT = 82, ASIN = 83, ACOS = 84, ATAN = 85, ATAN2 = 86, 
    LOG10 = 87, LOG = 88, EXP = 89, POW = 90, SQRT = 91, SQUARE = 92, SIGN = 93, 
    ROOT = 94, ROUND = 95, CEIL = 96, FLOOR = 97, LTRIM = 98, RTRIM = 99, 
    LOWER = 100, UPPER = 101, REVERSE = 102, LEN = 103, LEFT = 104, RIGHT = 105, 
    CONCAT = 106, CAST = 107, GEO_CONTAINS = 108, GEO_INTERSECT = 109, GEO_UNION = 110, 
    PLUS = 111, MINUS = 112, ASTERISK = 113, DIVISION = 114, MODULO = 115, 
    XOR = 116, EQUALS = 117, NOTEQUALS = 118, NOTEQUALS_GT_LT = 119, LPAREN = 120, 
    RPAREN = 121, GREATER = 122, LESS = 123, GREATEREQ = 124, LESSEQ = 125, 
    NOT = 126, OR = 127, AND = 128, BIT_OR = 129, BIT_AND = 130, L_SHIFT = 131, 
    R_SHIFT = 132, BOOLEANLIT = 133, TRUE = 134, FALSE = 135, FLOATLIT = 136, 
    INTLIT = 137, ID = 138
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



// Generated from /home/jvesely/dropdbase_instarea/dropdbase/GpuSqlParser/GpuSqlLexer.g4 by ANTLR 4.7.1

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
    GROUPBY = 49, AS = 50, IN = 51, ISNULL = 52, ISNOTNULL = 53, BETWEEN = 54, 
    ON = 55, ORDERBY = 56, DIR = 57, LIMIT = 58, OFFSET = 59, INNER = 60, 
    FULLOUTER = 61, SHOWDB = 62, SHOWTB = 63, SHOWCL = 64, AGG = 65, AVG = 66, 
    SUM = 67, MIN = 68, MAX = 69, COUNT = 70, YEAR = 71, MONTH = 72, DAY = 73, 
    HOUR = 74, MINUTE = 75, SECOND = 76, NOW = 77, PI = 78, ABS = 79, SIN = 80, 
    COS = 81, TAN = 82, COT = 83, ASIN = 84, ACOS = 85, ATAN = 86, ATAN2 = 87, 
    LOG10 = 88, LOG = 89, EXP = 90, POW = 91, SQRT = 92, SQUARE = 93, SIGN = 94, 
    ROOT = 95, ROUND = 96, CEIL = 97, FLOOR = 98, LTRIM = 99, RTRIM = 100, 
    LOWER = 101, UPPER = 102, REVERSE = 103, LEN = 104, LEFT = 105, RIGHT = 106, 
    CONCAT = 107, GEO_CONTAINS = 108, GEO_INTERSECT = 109, GEO_UNION = 110, 
    PLUS = 111, MINUS = 112, ASTERISK = 113, DIVISION = 114, MODULO = 115, 
    XOR = 116, EQUALS = 117, NOTEQUALS = 118, NOTEQUALS_GT_LT = 119, LPAREN = 120, 
    RPAREN = 121, GREATER = 122, LESS = 123, GREATEREQ = 124, LESSEQ = 125, 
    LOGICAL_NOT = 126, OR = 127, AND = 128, BIT_OR = 129, BIT_AND = 130, 
    L_SHIFT = 131, R_SHIFT = 132, BOOLEANLIT = 133, TRUE = 134, FALSE = 135, 
    FLOATLIT = 136, INTLIT = 137, NULLLIT = 138, ID = 139
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


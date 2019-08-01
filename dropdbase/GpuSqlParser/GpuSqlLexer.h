
// Generated from C:/Users/Andy/Desktop/parser\GpuSqlLexer.g4 by ANTLR 4.7.2

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
    AGG = 66, AVG = 67, SUM = 68, MIN = 69, MAX = 70, COUNT = 71, YEAR = 72, 
    MONTH = 73, DAY = 74, HOUR = 75, MINUTE = 76, SECOND = 77, NOW = 78, 
    PI = 79, ABS = 80, SIN = 81, COS = 82, TAN = 83, COT = 84, ASIN = 85, 
    ACOS = 86, ATAN = 87, ATAN2 = 88, LOG10 = 89, LOG = 90, EXP = 91, POW = 92, 
    SQRT = 93, SQUARE = 94, SIGN = 95, ROOT = 96, ROUND = 97, CEIL = 98, 
    FLOOR = 99, LTRIM = 100, RTRIM = 101, LOWER = 102, UPPER = 103, REVERSE = 104, 
    LEN = 105, LEFT = 106, RIGHT = 107, CONCAT = 108, CAST = 109, GEO_CONTAINS = 110, 
    GEO_INTERSECT = 111, GEO_UNION = 112, PLUS = 113, MINUS = 114, ASTERISK = 115, 
    DIVISION = 116, MODULO = 117, XOR = 118, EQUALS = 119, NOTEQUALS = 120, 
    NOTEQUALS_GT_LT = 121, LPAREN = 122, RPAREN = 123, GREATER = 124, LESS = 125, 
    GREATEREQ = 126, LESSEQ = 127, LOGICAL_NOT = 128, OR = 129, AND = 130, 
    BIT_OR = 131, BIT_AND = 132, L_SHIFT = 133, R_SHIFT = 134, BOOLEANLIT = 135, 
    TRUE = 136, FALSE = 137, FLOATLIT = 138, INTLIT = 139, NULLLIT = 140, 
    ID = 141
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



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
    POLYGON = 23, MULTIPOLYGON = 24, INTTYPE = 25, LONGTYPE = 26, DATETYPE = 27, 
    DETETIMETYPE = 28, FLOATTYPE = 29, DOUBLETYPE = 30, STRINGTYPE = 31, 
    BOOLEANTYPE = 32, POINTTYPE = 33, POLYTYPE = 34, INSERTINTO = 35, CREATEDB = 36, 
    DROPDB = 37, CREATETABLE = 38, DROPTABLE = 39, ALTERTABLE = 40, ALTERDATABASE = 41, 
    ADD = 42, DROPCOLUMN = 43, ALTERCOLUMN = 44, RENAMECOLUMN = 45, RENAMETO = 46, 
    CREATEINDEX = 47, INDEX = 48, PRIMARYKEY = 49, VALUES = 50, SELECT = 51, 
    FROM = 52, JOIN = 53, WHERE = 54, GROUPBY = 55, AS = 56, IN = 57, TO = 58, 
    ISNULL = 59, ISNOTNULL = 60, BETWEEN = 61, ON = 62, ORDERBY = 63, DIR = 64, 
    LIMIT = 65, OFFSET = 66, INNER = 67, FULLOUTER = 68, SHOWDB = 69, SHOWTB = 70, 
    SHOWCL = 71, AVG_AGG = 72, SUM_AGG = 73, MIN_AGG = 74, MAX_AGG = 75, 
    COUNT_AGG = 76, YEAR = 77, MONTH = 78, DAY = 79, HOUR = 80, MINUTE = 81, 
    SECOND = 82, NOW = 83, PI = 84, ABS = 85, SIN = 86, COS = 87, TAN = 88, 
    COT = 89, ASIN = 90, ACOS = 91, ATAN = 92, ATAN2 = 93, LOG10 = 94, LOG = 95, 
    EXP = 96, POW = 97, SQRT = 98, SQUARE = 99, SIGN = 100, ROOT = 101, 
    ROUND = 102, CEIL = 103, FLOOR = 104, LTRIM = 105, RTRIM = 106, LOWER = 107, 
    UPPER = 108, REVERSE = 109, LEN = 110, LEFT = 111, RIGHT = 112, CONCAT = 113, 
    CAST = 114, GEO_CONTAINS = 115, GEO_INTERSECT = 116, GEO_UNION = 117, 
    PLUS = 118, MINUS = 119, ASTERISK = 120, DIVISION = 121, MODULO = 122, 
    XOR = 123, EQUALS = 124, NOTEQUALS = 125, NOTEQUALS_GT_LT = 126, LPAREN = 127, 
    RPAREN = 128, GREATER = 129, LESS = 130, GREATEREQ = 131, LESSEQ = 132, 
    LOGICAL_NOT = 133, OR = 134, AND = 135, BIT_OR = 136, BIT_AND = 137, 
    L_SHIFT = 138, R_SHIFT = 139, BOOLEANLIT = 140, TRUE = 141, FALSE = 142, 
    FLOATLIT = 143, INTLIT = 144, NULLLIT = 145, ID = 146
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



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
    DROPCOLUMN = 42, ALTERCOLUMN = 43, RENAMECOLUMN = 44, RENAMETO = 45, 
    CREATEINDEX = 46, INDEX = 47, UNIQUE = 48, PRIMARYKEY = 49, CREATE = 50, 
    ADD = 51, DROP = 52, ALTER = 53, RENAME = 54, DATABASE = 55, TABLE = 56, 
    COLUMN = 57, VALUES = 58, SELECT = 59, FROM = 60, JOIN = 61, WHERE = 62, 
    GROUPBY = 63, AS = 64, IN = 65, TO = 66, ISNULL = 67, ISNOTNULL = 68, 
    BETWEEN = 69, ON = 70, ORDERBY = 71, DIR = 72, LIMIT = 73, OFFSET = 74, 
    INNER = 75, FULLOUTER = 76, SHOWDB = 77, SHOWTB = 78, SHOWCL = 79, AVG_AGG = 80, 
    SUM_AGG = 81, MIN_AGG = 82, MAX_AGG = 83, COUNT_AGG = 84, YEAR = 85, 
    MONTH = 86, DAY = 87, HOUR = 88, MINUTE = 89, SECOND = 90, NOW = 91, 
    PI = 92, ABS = 93, SIN = 94, COS = 95, TAN = 96, COT = 97, ASIN = 98, 
    ACOS = 99, ATAN = 100, ATAN2 = 101, LOG10 = 102, LOG = 103, EXP = 104, 
    POW = 105, SQRT = 106, SQUARE = 107, SIGN = 108, ROOT = 109, ROUND = 110, 
    CEIL = 111, FLOOR = 112, LTRIM = 113, RTRIM = 114, LOWER = 115, UPPER = 116, 
    REVERSE = 117, LEN = 118, LEFT = 119, RIGHT = 120, CONCAT = 121, CAST = 122, 
    GEO_CONTAINS = 123, GEO_INTERSECT = 124, GEO_UNION = 125, PLUS = 126, 
    MINUS = 127, ASTERISK = 128, DIVISION = 129, MODULO = 130, XOR = 131, 
    EQUALS = 132, NOTEQUALS = 133, NOTEQUALS_GT_LT = 134, LPAREN = 135, 
    RPAREN = 136, GREATER = 137, LESS = 138, GREATEREQ = 139, LESSEQ = 140, 
    LOGICAL_NOT = 141, OR = 142, AND = 143, BIT_OR = 144, BIT_AND = 145, 
    L_SHIFT = 146, R_SHIFT = 147, BOOLEANLIT = 148, TRUE = 149, FALSE = 150, 
    FLOATLIT = 151, INTLIT = 152, NULLLIT = 153, ID = 154
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


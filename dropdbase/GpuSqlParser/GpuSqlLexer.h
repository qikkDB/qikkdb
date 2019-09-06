
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
    NOTNULL = 69, BETWEEN = 70, ON = 71, ORDERBY = 72, DIR = 73, LIMIT = 74, 
    OFFSET = 75, INNER = 76, FULLOUTER = 77, SHOWDB = 78, SHOWTB = 79, SHOWCL = 80, 
    SHOWQTYPES = 81, AVG_AGG = 82, SUM_AGG = 83, MIN_AGG = 84, MAX_AGG = 85, 
    COUNT_AGG = 86, YEAR = 87, MONTH = 88, DAY = 89, HOUR = 90, MINUTE = 91, 
    SECOND = 92, NOW = 93, PI = 94, ABS = 95, SIN = 96, COS = 97, TAN = 98, 
    COT = 99, ASIN = 100, ACOS = 101, ATAN = 102, ATAN2 = 103, LOG10 = 104, 
    LOG = 105, EXP = 106, POW = 107, SQRT = 108, SQUARE = 109, SIGN = 110, 
    ROOT = 111, ROUND = 112, CEIL = 113, FLOOR = 114, LTRIM = 115, RTRIM = 116, 
    LOWER = 117, UPPER = 118, REVERSE = 119, LEN = 120, LEFT = 121, RIGHT = 122, 
    CONCAT = 123, CAST = 124, GEO_CONTAINS = 125, GEO_INTERSECT = 126, GEO_UNION = 127, 
    PLUS = 128, MINUS = 129, ASTERISK = 130, DIVISION = 131, MODULO = 132, 
    XOR = 133, EQUALS = 134, NOTEQUALS = 135, NOTEQUALS_GT_LT = 136, LPAREN = 137, 
    RPAREN = 138, GREATER = 139, LESS = 140, GREATEREQ = 141, LESSEQ = 142, 
    LOGICAL_NOT = 143, OR = 144, AND = 145, BIT_OR = 146, BIT_AND = 147, 
    L_SHIFT = 148, R_SHIFT = 149, BOOLEANLIT = 150, TRUE = 151, FALSE = 152, 
    FLOATLIT = 153, INTLIT = 154, NULLLIT = 155, ID = 156
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


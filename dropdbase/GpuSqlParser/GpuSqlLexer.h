
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
    AVG_AGG = 81, SUM_AGG = 82, MIN_AGG = 83, MAX_AGG = 84, COUNT_AGG = 85, 
    YEAR = 86, MONTH = 87, DAY = 88, HOUR = 89, MINUTE = 90, SECOND = 91, 
    NOW = 92, PI = 93, ABS = 94, SIN = 95, COS = 96, TAN = 97, COT = 98, 
    ASIN = 99, ACOS = 100, ATAN = 101, ATAN2 = 102, LOG10 = 103, LOG = 104, 
    EXP = 105, POW = 106, SQRT = 107, SQUARE = 108, SIGN = 109, ROOT = 110, 
    ROUND = 111, CEIL = 112, FLOOR = 113, LTRIM = 114, RTRIM = 115, LOWER = 116, 
    UPPER = 117, REVERSE = 118, LEN = 119, LEFT = 120, RIGHT = 121, CONCAT = 122, 
    CAST = 123, GEO_CONTAINS = 124, GEO_INTERSECT = 125, GEO_UNION = 126, 
    PLUS = 127, MINUS = 128, ASTERISK = 129, DIVISION = 130, MODULO = 131, 
    XOR = 132, EQUALS = 133, NOTEQUALS = 134, NOTEQUALS_GT_LT = 135, LPAREN = 136, 
    RPAREN = 137, GREATER = 138, LESS = 139, GREATEREQ = 140, LESSEQ = 141, 
    LOGICAL_NOT = 142, OR = 143, AND = 144, BIT_OR = 145, BIT_AND = 146, 
    L_SHIFT = 147, R_SHIFT = 148, BOOLEANLIT = 149, TRUE = 150, FALSE = 151, 
    FLOATLIT = 152, INTLIT = 153, NULLLIT = 154, ID = 155
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


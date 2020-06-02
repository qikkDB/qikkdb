
// Generated from C:/GPU-DB/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.8

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
    ADD = 51, DROP = 52, ALTER = 53, RENAME = 54, SET = 55, DATABASE = 56, 
    TABLE = 57, COLUMN = 58, BLOCKSIZE = 59, VALUES = 60, SELECT = 61, FROM = 62, 
    JOIN = 63, WHERE = 64, GROUPBY = 65, AS = 66, IN = 67, TO = 68, ISNULL = 69, 
    ISNOTNULL = 70, NOTNULL = 71, BETWEEN = 72, ON = 73, ORDERBY = 74, DIR = 75, 
    LIMIT = 76, OFFSET = 77, INNER = 78, FULLOUTER = 79, SHOWDB = 80, SHOWTB = 81, 
    SHOWCL = 82, SHOWQTYPES = 83, SHOWCONSTRAINTS = 84, AVG_AGG = 85, SUM_AGG = 86, 
    MIN_AGG = 87, MAX_AGG = 88, COUNT_AGG = 89, YEAR = 90, MONTH = 91, DAY = 92, 
    HOUR = 93, MINUTE = 94, SECOND = 95, WEEKDAY = 96, DAYOFWEEK = 97, NOW = 98, 
    PI = 99, ABS = 100, SIN = 101, COS = 102, TAN = 103, COT = 104, ASIN = 105, 
    ACOS = 106, ATAN = 107, ATAN2 = 108, LOG10 = 109, LOG = 110, EXP = 111, 
    POW = 112, SQRT = 113, SQUARE = 114, SIGN = 115, ROOT = 116, ROUND = 117, 
    CEIL = 118, FLOOR = 119, LTRIM = 120, RTRIM = 121, LOWER = 122, UPPER = 123, 
    REVERSE = 124, LEN = 125, LEFT = 126, RIGHT = 127, CONCAT = 128, CAST = 129, 
    RETPAYLOAD = 130, GEO_CONTAINS = 131, GEO_INTERSECT = 132, GEO_UNION = 133, 
    GEO_LONGITUDE_TO_TILE_X = 134, GEO_LATITUDE_TO_TILE_Y = 135, GEO_TILE_X_TO_LONGITUDE = 136, 
    GEO_TILE_Y_TO_LATITUDE = 137, PLUS = 138, MINUS = 139, ASTERISK = 140, 
    DIVISION = 141, MODULO = 142, XOR = 143, EQUALS = 144, NOTEQUALS = 145, 
    NOTEQUALS_GT_LT = 146, LPAREN = 147, RPAREN = 148, GREATER = 149, LESS = 150, 
    GREATEREQ = 151, LESSEQ = 152, LOGICAL_NOT = 153, OR = 154, AND = 155, 
    BIT_OR = 156, BIT_AND = 157, L_SHIFT = 158, R_SHIFT = 159, BOOLEANLIT = 160, 
    TRUE = 161, FALSE = 162, FLOATLIT = 163, INTLIT = 164, NULLLIT = 165, 
    ID = 166
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



// Generated from /Users/ms/dropdbase_instarea/dropdbase/GpuSqlParser/GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, UNDERSCORE = 9, COLON = 10, COMMA = 11, DOT = 12, DATELIT = 13, 
    TIMELIT = 14, DATATYPE = 15, POINT = 16, MULTIPOINT = 17, LINESTRING = 18, 
    MULTILINESTRING = 19, POLYGON = 20, MULTIPOLYGON = 21, INTTYPE = 22, 
    LONGTYPE = 23, FLOATTYPE = 24, DOUBLETYPE = 25, STRINGTYPE = 26, BOOLEANTYPE = 27, 
    POINTTYPE = 28, POLYTYPE = 29, INSERTINTO = 30, CREATEDB = 31, CREATETABLE = 32, 
    VALUES = 33, SELECT = 34, FROM = 35, JOIN = 36, WHERE = 37, GROUPBY = 38, 
    AS = 39, IN = 40, BETWEEN = 41, ON = 42, ORDERBY = 43, DIR = 44, LIMIT = 45, 
    OFFSET = 46, SHOWDB = 47, SHOWTB = 48, SHOWCL = 49, AGG = 50, AVG = 51, 
    SUM = 52, MIN = 53, MAX = 54, COUNT = 55, YEAR = 56, MONTH = 57, DAY = 58, 
    HOUR = 59, MINUTE = 60, SECOND = 61, GEO_CONTAINS = 62, GEO_INTERSECT = 63, 
    GEO_UNION = 64, PLUS = 65, MINUS = 66, ASTERISK = 67, DIVISION = 68, 
    MODULO = 69, EQUALS = 70, NOTEQUALS = 71, LPAREN = 72, RPAREN = 73, 
    GREATER = 74, LESS = 75, GREATEREQ = 76, LESSEQ = 77, NOT = 78, OR = 79, 
    AND = 80, FLOATLIT = 81, INTLIT = 82, ID = 83, BOOLEANLIT = 84, STRINGLIT = 85
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



// Generated from /Users/ms/dropdbase_instarea/dropdbase/GpuSqlParser/GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    LF = 1, CR = 2, CRLF = 3, WS = 4, SEMICOL = 5, COMMA = 6, DOT = 7, DATATYPE = 8, 
    POINT = 9, MULTIPOINT = 10, LINESTRING = 11, MULTILINESTRING = 12, POLYGON = 13, 
    MULTIPOLYGON = 14, INTTYPE = 15, LONGTYPE = 16, FLOATTYPE = 17, DOUBLETYPE = 18, 
    STRINGTYPE = 19, BOOLEANTYPE = 20, POINTTYPE = 21, POLYTYPE = 22, INSERTINTO = 23, 
    CREATEDB = 24, CREATETABLE = 25, VALUES = 26, SELECT = 27, FROM = 28, 
    JOIN = 29, WHERE = 30, GROUPBY = 31, AS = 32, IN = 33, BETWEEN = 34, 
    ON = 35, ORDERBY = 36, DIR = 37, LIMIT = 38, OFFSET = 39, SHOWDB = 40, 
    SHOWTB = 41, SHOWCL = 42, AGG = 43, AVG = 44, SUM = 45, MIN = 46, MAX = 47, 
    COUNT = 48, GEO = 49, CONTAINS = 50, PLUS = 51, MINUS = 52, ASTERISK = 53, 
    DIVISION = 54, MODULO = 55, EQUALS = 56, NOTEQUALS = 57, LPAREN = 58, 
    RPAREN = 59, GREATER = 60, LESS = 61, GREATEREQ = 62, LESSEQ = 63, NOT = 64, 
    OR = 65, AND = 66, FLOATLIT = 67, INTLIT = 68, ID = 69, BOOLEANLIT = 70, 
    STRINGLIT = 71
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


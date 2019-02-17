
// Generated from C:/Users/Martin Stano/Desktop/dropdbase_instarea/dropdbase/GpuSqlParser\GpuSqlLexer.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  GpuSqlLexer : public antlr4::Lexer {
public:
  enum {
    DATETIMELIT = 1, LF = 2, CR = 3, CRLF = 4, WS = 5, SEMICOL = 6, SQOUTE = 7, 
    DQOUTE = 8, COLON = 9, COMMA = 10, DOT = 11, DATELIT = 12, TIMELIT = 13, 
    DATATYPE = 14, POINT = 15, MULTIPOINT = 16, LINESTRING = 17, MULTILINESTRING = 18, 
    POLYGON = 19, MULTIPOLYGON = 20, INTTYPE = 21, LONGTYPE = 22, FLOATTYPE = 23, 
    DOUBLETYPE = 24, STRINGTYPE = 25, BOOLEANTYPE = 26, POINTTYPE = 27, 
    POLYTYPE = 28, INSERTINTO = 29, CREATEDB = 30, CREATETABLE = 31, VALUES = 32, 
    SELECT = 33, FROM = 34, JOIN = 35, WHERE = 36, GROUPBY = 37, AS = 38, 
    IN = 39, BETWEEN = 40, ON = 41, ORDERBY = 42, DIR = 43, LIMIT = 44, 
    OFFSET = 45, SHOWDB = 46, SHOWTB = 47, SHOWCL = 48, AGG = 49, AVG = 50, 
    SUM = 51, MIN = 52, MAX = 53, COUNT = 54, YEAR = 55, MONTH = 56, DAY = 57, 
    HOUR = 58, MINUTE = 59, SECOND = 60, GEO = 61, CONTAINS = 62, PLUS = 63, 
    MINUS = 64, ASTERISK = 65, DIVISION = 66, MODULO = 67, EQUALS = 68, 
    NOTEQUALS = 69, LPAREN = 70, RPAREN = 71, GREATER = 72, LESS = 73, GREATEREQ = 74, 
    LESSEQ = 75, NOT = 76, OR = 77, AND = 78, FLOATLIT = 79, INTLIT = 80, 
    ID = 81, BOOLEANLIT = 82, STRINGLIT = 83
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


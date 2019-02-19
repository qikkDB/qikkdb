
// Generated from C:/Users/Martin Stano/Desktop/dropdbase_instarea/dropdbase/GpuSqlParser\GpuSqlParser.g4 by ANTLR 4.7.2


#include "GpuSqlParserListener.h"

#include "GpuSqlParser.h"


using namespace antlrcpp;
using namespace antlr4;

GpuSqlParser::GpuSqlParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

GpuSqlParser::~GpuSqlParser() {
  delete _interpreter;
}

std::string GpuSqlParser::getGrammarFileName() const {
  return "GpuSqlParser.g4";
}

const std::vector<std::string>& GpuSqlParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& GpuSqlParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- SqlFileContext ------------------------------------------------------------------

GpuSqlParser::SqlFileContext::SqlFileContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlFileContext::EOF() {
  return getToken(GpuSqlParser::EOF, 0);
}

std::vector<GpuSqlParser::StatementContext *> GpuSqlParser::SqlFileContext::statement() {
  return getRuleContexts<GpuSqlParser::StatementContext>();
}

GpuSqlParser::StatementContext* GpuSqlParser::SqlFileContext::statement(size_t i) {
  return getRuleContext<GpuSqlParser::StatementContext>(i);
}


size_t GpuSqlParser::SqlFileContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlFile;
}

void GpuSqlParser::SqlFileContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlFile(this);
}

void GpuSqlParser::SqlFileContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlFile(this);
}

GpuSqlParser::SqlFileContext* GpuSqlParser::sqlFile() {
  SqlFileContext *_localctx = _tracker.createInstance<SqlFileContext>(_ctx, getState());
  enterRule(_localctx, 0, GpuSqlParser::RuleSqlFile);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(91);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << GpuSqlParser::INSERTINTO)
      | (1ULL << GpuSqlParser::CREATEDB)
      | (1ULL << GpuSqlParser::CREATETABLE)
      | (1ULL << GpuSqlParser::SELECT)
      | (1ULL << GpuSqlParser::SHOWDB)
      | (1ULL << GpuSqlParser::SHOWTB)
      | (1ULL << GpuSqlParser::SHOWCL))) != 0)) {
      setState(88);
      statement();
      setState(93);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(94);
    match(GpuSqlParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

GpuSqlParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::SqlSelectContext* GpuSqlParser::StatementContext::sqlSelect() {
  return getRuleContext<GpuSqlParser::SqlSelectContext>(0);
}

GpuSqlParser::SqlCreateDbContext* GpuSqlParser::StatementContext::sqlCreateDb() {
  return getRuleContext<GpuSqlParser::SqlCreateDbContext>(0);
}

GpuSqlParser::SqlCreateTableContext* GpuSqlParser::StatementContext::sqlCreateTable() {
  return getRuleContext<GpuSqlParser::SqlCreateTableContext>(0);
}

GpuSqlParser::SqlInsertIntoContext* GpuSqlParser::StatementContext::sqlInsertInto() {
  return getRuleContext<GpuSqlParser::SqlInsertIntoContext>(0);
}

GpuSqlParser::ShowStatementContext* GpuSqlParser::StatementContext::showStatement() {
  return getRuleContext<GpuSqlParser::ShowStatementContext>(0);
}


size_t GpuSqlParser::StatementContext::getRuleIndex() const {
  return GpuSqlParser::RuleStatement;
}

void GpuSqlParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void GpuSqlParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

GpuSqlParser::StatementContext* GpuSqlParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 2, GpuSqlParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(101);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SELECT: {
        enterOuterAlt(_localctx, 1);
        setState(96);
        sqlSelect();
        break;
      }

      case GpuSqlParser::CREATEDB: {
        enterOuterAlt(_localctx, 2);
        setState(97);
        sqlCreateDb();
        break;
      }

      case GpuSqlParser::CREATETABLE: {
        enterOuterAlt(_localctx, 3);
        setState(98);
        sqlCreateTable();
        break;
      }

      case GpuSqlParser::INSERTINTO: {
        enterOuterAlt(_localctx, 4);
        setState(99);
        sqlInsertInto();
        break;
      }

      case GpuSqlParser::SHOWDB:
      case GpuSqlParser::SHOWTB:
      case GpuSqlParser::SHOWCL: {
        enterOuterAlt(_localctx, 5);
        setState(100);
        showStatement();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShowStatementContext ------------------------------------------------------------------

GpuSqlParser::ShowStatementContext::ShowStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ShowDatabasesContext* GpuSqlParser::ShowStatementContext::showDatabases() {
  return getRuleContext<GpuSqlParser::ShowDatabasesContext>(0);
}

GpuSqlParser::ShowTablesContext* GpuSqlParser::ShowStatementContext::showTables() {
  return getRuleContext<GpuSqlParser::ShowTablesContext>(0);
}

GpuSqlParser::ShowColumnsContext* GpuSqlParser::ShowStatementContext::showColumns() {
  return getRuleContext<GpuSqlParser::ShowColumnsContext>(0);
}


size_t GpuSqlParser::ShowStatementContext::getRuleIndex() const {
  return GpuSqlParser::RuleShowStatement;
}

void GpuSqlParser::ShowStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShowStatement(this);
}

void GpuSqlParser::ShowStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShowStatement(this);
}

GpuSqlParser::ShowStatementContext* GpuSqlParser::showStatement() {
  ShowStatementContext *_localctx = _tracker.createInstance<ShowStatementContext>(_ctx, getState());
  enterRule(_localctx, 4, GpuSqlParser::RuleShowStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(106);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SHOWDB: {
        setState(103);
        showDatabases();
        break;
      }

      case GpuSqlParser::SHOWTB: {
        setState(104);
        showTables();
        break;
      }

      case GpuSqlParser::SHOWCL: {
        setState(105);
        showColumns();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShowDatabasesContext ------------------------------------------------------------------

GpuSqlParser::ShowDatabasesContext::ShowDatabasesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ShowDatabasesContext::SHOWDB() {
  return getToken(GpuSqlParser::SHOWDB, 0);
}

tree::TerminalNode* GpuSqlParser::ShowDatabasesContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::ShowDatabasesContext::getRuleIndex() const {
  return GpuSqlParser::RuleShowDatabases;
}

void GpuSqlParser::ShowDatabasesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShowDatabases(this);
}

void GpuSqlParser::ShowDatabasesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShowDatabases(this);
}

GpuSqlParser::ShowDatabasesContext* GpuSqlParser::showDatabases() {
  ShowDatabasesContext *_localctx = _tracker.createInstance<ShowDatabasesContext>(_ctx, getState());
  enterRule(_localctx, 6, GpuSqlParser::RuleShowDatabases);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(108);
    match(GpuSqlParser::SHOWDB);
    setState(109);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShowTablesContext ------------------------------------------------------------------

GpuSqlParser::ShowTablesContext::ShowTablesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ShowTablesContext::SHOWTB() {
  return getToken(GpuSqlParser::SHOWTB, 0);
}

tree::TerminalNode* GpuSqlParser::ShowTablesContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::ShowTablesContext::database() {
  return getRuleContext<GpuSqlParser::DatabaseContext>(0);
}

tree::TerminalNode* GpuSqlParser::ShowTablesContext::FROM() {
  return getToken(GpuSqlParser::FROM, 0);
}

tree::TerminalNode* GpuSqlParser::ShowTablesContext::IN() {
  return getToken(GpuSqlParser::IN, 0);
}


size_t GpuSqlParser::ShowTablesContext::getRuleIndex() const {
  return GpuSqlParser::RuleShowTables;
}

void GpuSqlParser::ShowTablesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShowTables(this);
}

void GpuSqlParser::ShowTablesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShowTables(this);
}

GpuSqlParser::ShowTablesContext* GpuSqlParser::showTables() {
  ShowTablesContext *_localctx = _tracker.createInstance<ShowTablesContext>(_ctx, getState());
  enterRule(_localctx, 8, GpuSqlParser::RuleShowTables);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(111);
    match(GpuSqlParser::SHOWTB);
    setState(114);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(112);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(113);
      database();
    }
    setState(116);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShowColumnsContext ------------------------------------------------------------------

GpuSqlParser::ShowColumnsContext::ShowColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ShowColumnsContext::SHOWCL() {
  return getToken(GpuSqlParser::SHOWCL, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::ShowColumnsContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

tree::TerminalNode* GpuSqlParser::ShowColumnsContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::ShowColumnsContext::FROM() {
  return getTokens(GpuSqlParser::FROM);
}

tree::TerminalNode* GpuSqlParser::ShowColumnsContext::FROM(size_t i) {
  return getToken(GpuSqlParser::FROM, i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::ShowColumnsContext::IN() {
  return getTokens(GpuSqlParser::IN);
}

tree::TerminalNode* GpuSqlParser::ShowColumnsContext::IN(size_t i) {
  return getToken(GpuSqlParser::IN, i);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::ShowColumnsContext::database() {
  return getRuleContext<GpuSqlParser::DatabaseContext>(0);
}


size_t GpuSqlParser::ShowColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleShowColumns;
}

void GpuSqlParser::ShowColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShowColumns(this);
}

void GpuSqlParser::ShowColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShowColumns(this);
}

GpuSqlParser::ShowColumnsContext* GpuSqlParser::showColumns() {
  ShowColumnsContext *_localctx = _tracker.createInstance<ShowColumnsContext>(_ctx, getState());
  enterRule(_localctx, 10, GpuSqlParser::RuleShowColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(118);
    match(GpuSqlParser::SHOWCL);
    setState(119);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(120);
    table();
    setState(123);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(121);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(122);
      database();
    }
    setState(125);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlSelectContext ------------------------------------------------------------------

GpuSqlParser::SqlSelectContext::SqlSelectContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::SELECT() {
  return getToken(GpuSqlParser::SELECT, 0);
}

GpuSqlParser::SelectColumnsContext* GpuSqlParser::SqlSelectContext::selectColumns() {
  return getRuleContext<GpuSqlParser::SelectColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::FROM() {
  return getToken(GpuSqlParser::FROM, 0);
}

GpuSqlParser::FromTablesContext* GpuSqlParser::SqlSelectContext::fromTables() {
  return getRuleContext<GpuSqlParser::FromTablesContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}

GpuSqlParser::JoinClausesContext* GpuSqlParser::SqlSelectContext::joinClauses() {
  return getRuleContext<GpuSqlParser::JoinClausesContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::WHERE() {
  return getToken(GpuSqlParser::WHERE, 0);
}

GpuSqlParser::WhereClauseContext* GpuSqlParser::SqlSelectContext::whereClause() {
  return getRuleContext<GpuSqlParser::WhereClauseContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::GROUPBY() {
  return getToken(GpuSqlParser::GROUPBY, 0);
}

GpuSqlParser::GroupByColumnsContext* GpuSqlParser::SqlSelectContext::groupByColumns() {
  return getRuleContext<GpuSqlParser::GroupByColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::ORDERBY() {
  return getToken(GpuSqlParser::ORDERBY, 0);
}

GpuSqlParser::OrderByColumnsContext* GpuSqlParser::SqlSelectContext::orderByColumns() {
  return getRuleContext<GpuSqlParser::OrderByColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::LIMIT() {
  return getToken(GpuSqlParser::LIMIT, 0);
}

GpuSqlParser::LimitContext* GpuSqlParser::SqlSelectContext::limit() {
  return getRuleContext<GpuSqlParser::LimitContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::OFFSET() {
  return getToken(GpuSqlParser::OFFSET, 0);
}

GpuSqlParser::OffsetContext* GpuSqlParser::SqlSelectContext::offset() {
  return getRuleContext<GpuSqlParser::OffsetContext>(0);
}


size_t GpuSqlParser::SqlSelectContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlSelect;
}

void GpuSqlParser::SqlSelectContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlSelect(this);
}

void GpuSqlParser::SqlSelectContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlSelect(this);
}

GpuSqlParser::SqlSelectContext* GpuSqlParser::sqlSelect() {
  SqlSelectContext *_localctx = _tracker.createInstance<SqlSelectContext>(_ctx, getState());
  enterRule(_localctx, 12, GpuSqlParser::RuleSqlSelect);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(127);
    match(GpuSqlParser::SELECT);
    setState(128);
    selectColumns();
    setState(129);
    match(GpuSqlParser::FROM);
    setState(130);
    fromTables();
    setState(132);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::JOIN) {
      setState(131);
      joinClauses();
    }
    setState(136);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::WHERE) {
      setState(134);
      match(GpuSqlParser::WHERE);
      setState(135);
      whereClause();
    }
    setState(140);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::GROUPBY) {
      setState(138);
      match(GpuSqlParser::GROUPBY);
      setState(139);
      groupByColumns();
    }
    setState(144);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::ORDERBY) {
      setState(142);
      match(GpuSqlParser::ORDERBY);
      setState(143);
      orderByColumns();
    }
    setState(148);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::LIMIT) {
      setState(146);
      match(GpuSqlParser::LIMIT);
      setState(147);
      limit();
    }
    setState(152);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::OFFSET) {
      setState(150);
      match(GpuSqlParser::OFFSET);
      setState(151);
      offset();
    }
    setState(154);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlCreateDbContext ------------------------------------------------------------------

GpuSqlParser::SqlCreateDbContext::SqlCreateDbContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlCreateDbContext::CREATEDB() {
  return getToken(GpuSqlParser::CREATEDB, 0);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::SqlCreateDbContext::database() {
  return getRuleContext<GpuSqlParser::DatabaseContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateDbContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::SqlCreateDbContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlCreateDb;
}

void GpuSqlParser::SqlCreateDbContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlCreateDb(this);
}

void GpuSqlParser::SqlCreateDbContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlCreateDb(this);
}

GpuSqlParser::SqlCreateDbContext* GpuSqlParser::sqlCreateDb() {
  SqlCreateDbContext *_localctx = _tracker.createInstance<SqlCreateDbContext>(_ctx, getState());
  enterRule(_localctx, 14, GpuSqlParser::RuleSqlCreateDb);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(156);
    match(GpuSqlParser::CREATEDB);
    setState(157);
    database();
    setState(158);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlCreateTableContext ------------------------------------------------------------------

GpuSqlParser::SqlCreateTableContext::SqlCreateTableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlCreateTableContext::CREATETABLE() {
  return getToken(GpuSqlParser::CREATETABLE, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::SqlCreateTableContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateTableContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::NewTableColumnsContext* GpuSqlParser::SqlCreateTableContext::newTableColumns() {
  return getRuleContext<GpuSqlParser::NewTableColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateTableContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}


size_t GpuSqlParser::SqlCreateTableContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlCreateTable;
}

void GpuSqlParser::SqlCreateTableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlCreateTable(this);
}

void GpuSqlParser::SqlCreateTableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlCreateTable(this);
}

GpuSqlParser::SqlCreateTableContext* GpuSqlParser::sqlCreateTable() {
  SqlCreateTableContext *_localctx = _tracker.createInstance<SqlCreateTableContext>(_ctx, getState());
  enterRule(_localctx, 16, GpuSqlParser::RuleSqlCreateTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(160);
    match(GpuSqlParser::CREATETABLE);
    setState(161);
    table();
    setState(162);
    match(GpuSqlParser::LPAREN);
    setState(163);
    newTableColumns();
    setState(164);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlInsertIntoContext ------------------------------------------------------------------

GpuSqlParser::SqlInsertIntoContext::SqlInsertIntoContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlInsertIntoContext::INSERTINTO() {
  return getToken(GpuSqlParser::INSERTINTO, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::SqlInsertIntoContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::SqlInsertIntoContext::LPAREN() {
  return getTokens(GpuSqlParser::LPAREN);
}

tree::TerminalNode* GpuSqlParser::SqlInsertIntoContext::LPAREN(size_t i) {
  return getToken(GpuSqlParser::LPAREN, i);
}

GpuSqlParser::InsertIntoColumnsContext* GpuSqlParser::SqlInsertIntoContext::insertIntoColumns() {
  return getRuleContext<GpuSqlParser::InsertIntoColumnsContext>(0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::SqlInsertIntoContext::RPAREN() {
  return getTokens(GpuSqlParser::RPAREN);
}

tree::TerminalNode* GpuSqlParser::SqlInsertIntoContext::RPAREN(size_t i) {
  return getToken(GpuSqlParser::RPAREN, i);
}

tree::TerminalNode* GpuSqlParser::SqlInsertIntoContext::VALUES() {
  return getToken(GpuSqlParser::VALUES, 0);
}

GpuSqlParser::InsertIntoValuesContext* GpuSqlParser::SqlInsertIntoContext::insertIntoValues() {
  return getRuleContext<GpuSqlParser::InsertIntoValuesContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlInsertIntoContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::SqlInsertIntoContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlInsertInto;
}

void GpuSqlParser::SqlInsertIntoContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlInsertInto(this);
}

void GpuSqlParser::SqlInsertIntoContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlInsertInto(this);
}

GpuSqlParser::SqlInsertIntoContext* GpuSqlParser::sqlInsertInto() {
  SqlInsertIntoContext *_localctx = _tracker.createInstance<SqlInsertIntoContext>(_ctx, getState());
  enterRule(_localctx, 18, GpuSqlParser::RuleSqlInsertInto);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(166);
    match(GpuSqlParser::INSERTINTO);
    setState(167);
    table();
    setState(168);
    match(GpuSqlParser::LPAREN);
    setState(169);
    insertIntoColumns();
    setState(170);
    match(GpuSqlParser::RPAREN);
    setState(171);
    match(GpuSqlParser::VALUES);
    setState(172);
    match(GpuSqlParser::LPAREN);
    setState(173);
    insertIntoValues();
    setState(174);
    match(GpuSqlParser::RPAREN);
    setState(175);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NewTableColumnsContext ------------------------------------------------------------------

GpuSqlParser::NewTableColumnsContext::NewTableColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::NewTableColumnContext *> GpuSqlParser::NewTableColumnsContext::newTableColumn() {
  return getRuleContexts<GpuSqlParser::NewTableColumnContext>();
}

GpuSqlParser::NewTableColumnContext* GpuSqlParser::NewTableColumnsContext::newTableColumn(size_t i) {
  return getRuleContext<GpuSqlParser::NewTableColumnContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::NewTableColumnsContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::NewTableColumnsContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::NewTableColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleNewTableColumns;
}

void GpuSqlParser::NewTableColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNewTableColumns(this);
}

void GpuSqlParser::NewTableColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNewTableColumns(this);
}

GpuSqlParser::NewTableColumnsContext* GpuSqlParser::newTableColumns() {
  NewTableColumnsContext *_localctx = _tracker.createInstance<NewTableColumnsContext>(_ctx, getState());
  enterRule(_localctx, 20, GpuSqlParser::RuleNewTableColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(177);
    newTableColumn();
    setState(182);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(178);
      match(GpuSqlParser::COMMA);
      setState(179);
      newTableColumn();
      setState(184);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NewTableColumnContext ------------------------------------------------------------------

GpuSqlParser::NewTableColumnContext::NewTableColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ColumnIdContext* GpuSqlParser::NewTableColumnContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
}

tree::TerminalNode* GpuSqlParser::NewTableColumnContext::DATATYPE() {
  return getToken(GpuSqlParser::DATATYPE, 0);
}


size_t GpuSqlParser::NewTableColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleNewTableColumn;
}

void GpuSqlParser::NewTableColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNewTableColumn(this);
}

void GpuSqlParser::NewTableColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNewTableColumn(this);
}

GpuSqlParser::NewTableColumnContext* GpuSqlParser::newTableColumn() {
  NewTableColumnContext *_localctx = _tracker.createInstance<NewTableColumnContext>(_ctx, getState());
  enterRule(_localctx, 22, GpuSqlParser::RuleNewTableColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(185);
    columnId();
    setState(186);
    match(GpuSqlParser::DATATYPE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SelectColumnsContext ------------------------------------------------------------------

GpuSqlParser::SelectColumnsContext::SelectColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::SelectColumnContext *> GpuSqlParser::SelectColumnsContext::selectColumn() {
  return getRuleContexts<GpuSqlParser::SelectColumnContext>();
}

GpuSqlParser::SelectColumnContext* GpuSqlParser::SelectColumnsContext::selectColumn(size_t i) {
  return getRuleContext<GpuSqlParser::SelectColumnContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::SelectColumnsContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::SelectColumnsContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::SelectColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleSelectColumns;
}

void GpuSqlParser::SelectColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSelectColumns(this);
}

void GpuSqlParser::SelectColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSelectColumns(this);
}

GpuSqlParser::SelectColumnsContext* GpuSqlParser::selectColumns() {
  SelectColumnsContext *_localctx = _tracker.createInstance<SelectColumnsContext>(_ctx, getState());
  enterRule(_localctx, 24, GpuSqlParser::RuleSelectColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(188);
    selectColumn();
    setState(193);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(189);
      match(GpuSqlParser::COMMA);
      setState(190);
      selectColumn();
      setState(195);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SelectColumnContext ------------------------------------------------------------------

GpuSqlParser::SelectColumnContext::SelectColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ExpressionContext* GpuSqlParser::SelectColumnContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}


size_t GpuSqlParser::SelectColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleSelectColumn;
}

void GpuSqlParser::SelectColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSelectColumn(this);
}

void GpuSqlParser::SelectColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSelectColumn(this);
}

GpuSqlParser::SelectColumnContext* GpuSqlParser::selectColumn() {
  SelectColumnContext *_localctx = _tracker.createInstance<SelectColumnContext>(_ctx, getState());
  enterRule(_localctx, 26, GpuSqlParser::RuleSelectColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(196);
    expression(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WhereClauseContext ------------------------------------------------------------------

GpuSqlParser::WhereClauseContext::WhereClauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ExpressionContext* GpuSqlParser::WhereClauseContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}


size_t GpuSqlParser::WhereClauseContext::getRuleIndex() const {
  return GpuSqlParser::RuleWhereClause;
}

void GpuSqlParser::WhereClauseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhereClause(this);
}

void GpuSqlParser::WhereClauseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhereClause(this);
}

GpuSqlParser::WhereClauseContext* GpuSqlParser::whereClause() {
  WhereClauseContext *_localctx = _tracker.createInstance<WhereClauseContext>(_ctx, getState());
  enterRule(_localctx, 28, GpuSqlParser::RuleWhereClause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(198);
    expression(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OrderByColumnsContext ------------------------------------------------------------------

GpuSqlParser::OrderByColumnsContext::OrderByColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::OrderByColumnContext *> GpuSqlParser::OrderByColumnsContext::orderByColumn() {
  return getRuleContexts<GpuSqlParser::OrderByColumnContext>();
}

GpuSqlParser::OrderByColumnContext* GpuSqlParser::OrderByColumnsContext::orderByColumn(size_t i) {
  return getRuleContext<GpuSqlParser::OrderByColumnContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::OrderByColumnsContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::OrderByColumnsContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::OrderByColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleOrderByColumns;
}

void GpuSqlParser::OrderByColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOrderByColumns(this);
}

void GpuSqlParser::OrderByColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOrderByColumns(this);
}

GpuSqlParser::OrderByColumnsContext* GpuSqlParser::orderByColumns() {
  OrderByColumnsContext *_localctx = _tracker.createInstance<OrderByColumnsContext>(_ctx, getState());
  enterRule(_localctx, 30, GpuSqlParser::RuleOrderByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(200);
    orderByColumn();
    setState(205);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(201);
      match(GpuSqlParser::COMMA);
      setState(202);
      orderByColumn();
      setState(207);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OrderByColumnContext ------------------------------------------------------------------

GpuSqlParser::OrderByColumnContext::OrderByColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ColumnIdContext* GpuSqlParser::OrderByColumnContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
}

tree::TerminalNode* GpuSqlParser::OrderByColumnContext::DIR() {
  return getToken(GpuSqlParser::DIR, 0);
}


size_t GpuSqlParser::OrderByColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleOrderByColumn;
}

void GpuSqlParser::OrderByColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOrderByColumn(this);
}

void GpuSqlParser::OrderByColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOrderByColumn(this);
}

GpuSqlParser::OrderByColumnContext* GpuSqlParser::orderByColumn() {
  OrderByColumnContext *_localctx = _tracker.createInstance<OrderByColumnContext>(_ctx, getState());
  enterRule(_localctx, 32, GpuSqlParser::RuleOrderByColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(208);
    columnId();
    setState(210);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::DIR) {
      setState(209);
      match(GpuSqlParser::DIR);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InsertIntoValuesContext ------------------------------------------------------------------

GpuSqlParser::InsertIntoValuesContext::InsertIntoValuesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::ColumnValueContext *> GpuSqlParser::InsertIntoValuesContext::columnValue() {
  return getRuleContexts<GpuSqlParser::ColumnValueContext>();
}

GpuSqlParser::ColumnValueContext* GpuSqlParser::InsertIntoValuesContext::columnValue(size_t i) {
  return getRuleContext<GpuSqlParser::ColumnValueContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::InsertIntoValuesContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::InsertIntoValuesContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::InsertIntoValuesContext::getRuleIndex() const {
  return GpuSqlParser::RuleInsertIntoValues;
}

void GpuSqlParser::InsertIntoValuesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInsertIntoValues(this);
}

void GpuSqlParser::InsertIntoValuesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInsertIntoValues(this);
}

GpuSqlParser::InsertIntoValuesContext* GpuSqlParser::insertIntoValues() {
  InsertIntoValuesContext *_localctx = _tracker.createInstance<InsertIntoValuesContext>(_ctx, getState());
  enterRule(_localctx, 34, GpuSqlParser::RuleInsertIntoValues);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(212);
    columnValue();
    setState(217);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(213);
      match(GpuSqlParser::COMMA);
      setState(214);
      columnValue();
      setState(219);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InsertIntoColumnsContext ------------------------------------------------------------------

GpuSqlParser::InsertIntoColumnsContext::InsertIntoColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::ColumnIdContext *> GpuSqlParser::InsertIntoColumnsContext::columnId() {
  return getRuleContexts<GpuSqlParser::ColumnIdContext>();
}

GpuSqlParser::ColumnIdContext* GpuSqlParser::InsertIntoColumnsContext::columnId(size_t i) {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::InsertIntoColumnsContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::InsertIntoColumnsContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::InsertIntoColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleInsertIntoColumns;
}

void GpuSqlParser::InsertIntoColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInsertIntoColumns(this);
}

void GpuSqlParser::InsertIntoColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInsertIntoColumns(this);
}

GpuSqlParser::InsertIntoColumnsContext* GpuSqlParser::insertIntoColumns() {
  InsertIntoColumnsContext *_localctx = _tracker.createInstance<InsertIntoColumnsContext>(_ctx, getState());
  enterRule(_localctx, 36, GpuSqlParser::RuleInsertIntoColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(220);
    columnId();
    setState(225);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(221);
      match(GpuSqlParser::COMMA);
      setState(222);
      columnId();
      setState(227);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GroupByColumnsContext ------------------------------------------------------------------

GpuSqlParser::GroupByColumnsContext::GroupByColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::GroupByColumnContext *> GpuSqlParser::GroupByColumnsContext::groupByColumn() {
  return getRuleContexts<GpuSqlParser::GroupByColumnContext>();
}

GpuSqlParser::GroupByColumnContext* GpuSqlParser::GroupByColumnsContext::groupByColumn(size_t i) {
  return getRuleContext<GpuSqlParser::GroupByColumnContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::GroupByColumnsContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::GroupByColumnsContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::GroupByColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleGroupByColumns;
}

void GpuSqlParser::GroupByColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGroupByColumns(this);
}

void GpuSqlParser::GroupByColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGroupByColumns(this);
}

GpuSqlParser::GroupByColumnsContext* GpuSqlParser::groupByColumns() {
  GroupByColumnsContext *_localctx = _tracker.createInstance<GroupByColumnsContext>(_ctx, getState());
  enterRule(_localctx, 38, GpuSqlParser::RuleGroupByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(228);
    groupByColumn();
    setState(233);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(229);
      match(GpuSqlParser::COMMA);
      setState(230);
      groupByColumn();
      setState(235);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GroupByColumnContext ------------------------------------------------------------------

GpuSqlParser::GroupByColumnContext::GroupByColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ExpressionContext* GpuSqlParser::GroupByColumnContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}


size_t GpuSqlParser::GroupByColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleGroupByColumn;
}

void GpuSqlParser::GroupByColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGroupByColumn(this);
}

void GpuSqlParser::GroupByColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGroupByColumn(this);
}

GpuSqlParser::GroupByColumnContext* GpuSqlParser::groupByColumn() {
  GroupByColumnContext *_localctx = _tracker.createInstance<GroupByColumnContext>(_ctx, getState());
  enterRule(_localctx, 40, GpuSqlParser::RuleGroupByColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(236);
    expression(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ColumnIdContext ------------------------------------------------------------------

GpuSqlParser::ColumnIdContext::ColumnIdContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ColumnContext* GpuSqlParser::ColumnIdContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
}

GpuSqlParser::TableContext* GpuSqlParser::ColumnIdContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

tree::TerminalNode* GpuSqlParser::ColumnIdContext::DOT() {
  return getToken(GpuSqlParser::DOT, 0);
}


size_t GpuSqlParser::ColumnIdContext::getRuleIndex() const {
  return GpuSqlParser::RuleColumnId;
}

void GpuSqlParser::ColumnIdContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterColumnId(this);
}

void GpuSqlParser::ColumnIdContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitColumnId(this);
}

GpuSqlParser::ColumnIdContext* GpuSqlParser::columnId() {
  ColumnIdContext *_localctx = _tracker.createInstance<ColumnIdContext>(_ctx, getState());
  enterRule(_localctx, 42, GpuSqlParser::RuleColumnId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(243);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(238);
      column();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(239);
      table();
      setState(240);
      match(GpuSqlParser::DOT);
      setState(241);
      column();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FromTablesContext ------------------------------------------------------------------

GpuSqlParser::FromTablesContext::FromTablesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::TableContext *> GpuSqlParser::FromTablesContext::table() {
  return getRuleContexts<GpuSqlParser::TableContext>();
}

GpuSqlParser::TableContext* GpuSqlParser::FromTablesContext::table(size_t i) {
  return getRuleContext<GpuSqlParser::TableContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::FromTablesContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::FromTablesContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::FromTablesContext::getRuleIndex() const {
  return GpuSqlParser::RuleFromTables;
}

void GpuSqlParser::FromTablesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFromTables(this);
}

void GpuSqlParser::FromTablesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFromTables(this);
}

GpuSqlParser::FromTablesContext* GpuSqlParser::fromTables() {
  FromTablesContext *_localctx = _tracker.createInstance<FromTablesContext>(_ctx, getState());
  enterRule(_localctx, 44, GpuSqlParser::RuleFromTables);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(245);
    table();
    setState(250);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(246);
      match(GpuSqlParser::COMMA);
      setState(247);
      table();
      setState(252);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- JoinClausesContext ------------------------------------------------------------------

GpuSqlParser::JoinClausesContext::JoinClausesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::JoinClauseContext *> GpuSqlParser::JoinClausesContext::joinClause() {
  return getRuleContexts<GpuSqlParser::JoinClauseContext>();
}

GpuSqlParser::JoinClauseContext* GpuSqlParser::JoinClausesContext::joinClause(size_t i) {
  return getRuleContext<GpuSqlParser::JoinClauseContext>(i);
}


size_t GpuSqlParser::JoinClausesContext::getRuleIndex() const {
  return GpuSqlParser::RuleJoinClauses;
}

void GpuSqlParser::JoinClausesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterJoinClauses(this);
}

void GpuSqlParser::JoinClausesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitJoinClauses(this);
}

GpuSqlParser::JoinClausesContext* GpuSqlParser::joinClauses() {
  JoinClausesContext *_localctx = _tracker.createInstance<JoinClausesContext>(_ctx, getState());
  enterRule(_localctx, 46, GpuSqlParser::RuleJoinClauses);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(254); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(253);
      joinClause();
      setState(256); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == GpuSqlParser::JOIN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- JoinClauseContext ------------------------------------------------------------------

GpuSqlParser::JoinClauseContext::JoinClauseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::JoinClauseContext::JOIN() {
  return getToken(GpuSqlParser::JOIN, 0);
}

GpuSqlParser::JoinTableContext* GpuSqlParser::JoinClauseContext::joinTable() {
  return getRuleContext<GpuSqlParser::JoinTableContext>(0);
}

tree::TerminalNode* GpuSqlParser::JoinClauseContext::ON() {
  return getToken(GpuSqlParser::ON, 0);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::JoinClauseContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}


size_t GpuSqlParser::JoinClauseContext::getRuleIndex() const {
  return GpuSqlParser::RuleJoinClause;
}

void GpuSqlParser::JoinClauseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterJoinClause(this);
}

void GpuSqlParser::JoinClauseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitJoinClause(this);
}

GpuSqlParser::JoinClauseContext* GpuSqlParser::joinClause() {
  JoinClauseContext *_localctx = _tracker.createInstance<JoinClauseContext>(_ctx, getState());
  enterRule(_localctx, 48, GpuSqlParser::RuleJoinClause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(258);
    match(GpuSqlParser::JOIN);
    setState(259);
    joinTable();
    setState(260);
    match(GpuSqlParser::ON);
    setState(261);
    expression(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- JoinTableContext ------------------------------------------------------------------

GpuSqlParser::JoinTableContext::JoinTableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::TableContext* GpuSqlParser::JoinTableContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}


size_t GpuSqlParser::JoinTableContext::getRuleIndex() const {
  return GpuSqlParser::RuleJoinTable;
}

void GpuSqlParser::JoinTableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterJoinTable(this);
}

void GpuSqlParser::JoinTableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitJoinTable(this);
}

GpuSqlParser::JoinTableContext* GpuSqlParser::joinTable() {
  JoinTableContext *_localctx = _tracker.createInstance<JoinTableContext>(_ctx, getState());
  enterRule(_localctx, 50, GpuSqlParser::RuleJoinTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(263);
    table();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TableContext ------------------------------------------------------------------

GpuSqlParser::TableContext::TableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::TableContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}


size_t GpuSqlParser::TableContext::getRuleIndex() const {
  return GpuSqlParser::RuleTable;
}

void GpuSqlParser::TableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTable(this);
}

void GpuSqlParser::TableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTable(this);
}

GpuSqlParser::TableContext* GpuSqlParser::table() {
  TableContext *_localctx = _tracker.createInstance<TableContext>(_ctx, getState());
  enterRule(_localctx, 52, GpuSqlParser::RuleTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(265);
    match(GpuSqlParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ColumnContext ------------------------------------------------------------------

GpuSqlParser::ColumnContext::ColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ColumnContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}


size_t GpuSqlParser::ColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleColumn;
}

void GpuSqlParser::ColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterColumn(this);
}

void GpuSqlParser::ColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitColumn(this);
}

GpuSqlParser::ColumnContext* GpuSqlParser::column() {
  ColumnContext *_localctx = _tracker.createInstance<ColumnContext>(_ctx, getState());
  enterRule(_localctx, 54, GpuSqlParser::RuleColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(267);
    match(GpuSqlParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DatabaseContext ------------------------------------------------------------------

GpuSqlParser::DatabaseContext::DatabaseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::DatabaseContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}


size_t GpuSqlParser::DatabaseContext::getRuleIndex() const {
  return GpuSqlParser::RuleDatabase;
}

void GpuSqlParser::DatabaseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDatabase(this);
}

void GpuSqlParser::DatabaseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDatabase(this);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::database() {
  DatabaseContext *_localctx = _tracker.createInstance<DatabaseContext>(_ctx, getState());
  enterRule(_localctx, 56, GpuSqlParser::RuleDatabase);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(269);
    match(GpuSqlParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LimitContext ------------------------------------------------------------------

GpuSqlParser::LimitContext::LimitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::LimitContext::INTLIT() {
  return getToken(GpuSqlParser::INTLIT, 0);
}


size_t GpuSqlParser::LimitContext::getRuleIndex() const {
  return GpuSqlParser::RuleLimit;
}

void GpuSqlParser::LimitContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLimit(this);
}

void GpuSqlParser::LimitContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLimit(this);
}

GpuSqlParser::LimitContext* GpuSqlParser::limit() {
  LimitContext *_localctx = _tracker.createInstance<LimitContext>(_ctx, getState());
  enterRule(_localctx, 58, GpuSqlParser::RuleLimit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(271);
    match(GpuSqlParser::INTLIT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OffsetContext ------------------------------------------------------------------

GpuSqlParser::OffsetContext::OffsetContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::OffsetContext::INTLIT() {
  return getToken(GpuSqlParser::INTLIT, 0);
}


size_t GpuSqlParser::OffsetContext::getRuleIndex() const {
  return GpuSqlParser::RuleOffset;
}

void GpuSqlParser::OffsetContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOffset(this);
}

void GpuSqlParser::OffsetContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOffset(this);
}

GpuSqlParser::OffsetContext* GpuSqlParser::offset() {
  OffsetContext *_localctx = _tracker.createInstance<OffsetContext>(_ctx, getState());
  enterRule(_localctx, 60, GpuSqlParser::RuleOffset);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(273);
    match(GpuSqlParser::INTLIT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ColumnValueContext ------------------------------------------------------------------

GpuSqlParser::ColumnValueContext::ColumnValueContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ColumnValueContext::INTLIT() {
  return getToken(GpuSqlParser::INTLIT, 0);
}

tree::TerminalNode* GpuSqlParser::ColumnValueContext::FLOATLIT() {
  return getToken(GpuSqlParser::FLOATLIT, 0);
}

GpuSqlParser::GeometryContext* GpuSqlParser::ColumnValueContext::geometry() {
  return getRuleContext<GpuSqlParser::GeometryContext>(0);
}

tree::TerminalNode* GpuSqlParser::ColumnValueContext::STRINGLIT() {
  return getToken(GpuSqlParser::STRINGLIT, 0);
}


size_t GpuSqlParser::ColumnValueContext::getRuleIndex() const {
  return GpuSqlParser::RuleColumnValue;
}

void GpuSqlParser::ColumnValueContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterColumnValue(this);
}

void GpuSqlParser::ColumnValueContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitColumnValue(this);
}

GpuSqlParser::ColumnValueContext* GpuSqlParser::columnValue() {
  ColumnValueContext *_localctx = _tracker.createInstance<ColumnValueContext>(_ctx, getState());
  enterRule(_localctx, 62, GpuSqlParser::RuleColumnValue);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(280);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::INTLIT: {
        setState(275);
        match(GpuSqlParser::INTLIT);
        break;
      }

      case GpuSqlParser::FLOATLIT: {
        setState(276);
        match(GpuSqlParser::FLOATLIT);
        break;
      }

      case GpuSqlParser::POINT:
      case GpuSqlParser::MULTIPOINT:
      case GpuSqlParser::LINESTRING:
      case GpuSqlParser::MULTILINESTRING:
      case GpuSqlParser::POLYGON:
      case GpuSqlParser::MULTIPOLYGON: {
        setState(277);
        geometry();
        break;
      }

      case GpuSqlParser::STRINGLIT: {
        setState(278);
        match(GpuSqlParser::STRINGLIT);
        break;
      }

      case GpuSqlParser::COMMA:
      case GpuSqlParser::RPAREN: {
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

GpuSqlParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t GpuSqlParser::ExpressionContext::getRuleIndex() const {
  return GpuSqlParser::RuleExpression;
}

void GpuSqlParser::ExpressionContext::copyFrom(ExpressionContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- DecimalLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::DecimalLiteralContext::FLOATLIT() {
  return getToken(GpuSqlParser::FLOATLIT, 0);
}

GpuSqlParser::DecimalLiteralContext::DecimalLiteralContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::DecimalLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecimalLiteral(this);
}
void GpuSqlParser::DecimalLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecimalLiteral(this);
}
//----------------- GeoReferenceContext ------------------------------------------------------------------

GpuSqlParser::GeometryContext* GpuSqlParser::GeoReferenceContext::geometry() {
  return getRuleContext<GpuSqlParser::GeometryContext>(0);
}

GpuSqlParser::GeoReferenceContext::GeoReferenceContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::GeoReferenceContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGeoReference(this);
}
void GpuSqlParser::GeoReferenceContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGeoReference(this);
}
//----------------- DateTimeLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::DateTimeLiteralContext::DATETIMELIT() {
  return getToken(GpuSqlParser::DATETIMELIT, 0);
}

GpuSqlParser::DateTimeLiteralContext::DateTimeLiteralContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::DateTimeLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDateTimeLiteral(this);
}
void GpuSqlParser::DateTimeLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDateTimeLiteral(this);
}
//----------------- StringLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::StringLiteralContext::STRINGLIT() {
  return getToken(GpuSqlParser::STRINGLIT, 0);
}

GpuSqlParser::StringLiteralContext::StringLiteralContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::StringLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStringLiteral(this);
}
void GpuSqlParser::StringLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStringLiteral(this);
}
//----------------- IntLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::IntLiteralContext::INTLIT() {
  return getToken(GpuSqlParser::INTLIT, 0);
}

GpuSqlParser::IntLiteralContext::IntLiteralContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::IntLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIntLiteral(this);
}
void GpuSqlParser::IntLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIntLiteral(this);
}
//----------------- TernaryOperationContext ------------------------------------------------------------------

std::vector<GpuSqlParser::ExpressionContext *> GpuSqlParser::TernaryOperationContext::expression() {
  return getRuleContexts<GpuSqlParser::ExpressionContext>();
}

GpuSqlParser::ExpressionContext* GpuSqlParser::TernaryOperationContext::expression(size_t i) {
  return getRuleContext<GpuSqlParser::ExpressionContext>(i);
}

tree::TerminalNode* GpuSqlParser::TernaryOperationContext::BETWEEN() {
  return getToken(GpuSqlParser::BETWEEN, 0);
}

tree::TerminalNode* GpuSqlParser::TernaryOperationContext::AND() {
  return getToken(GpuSqlParser::AND, 0);
}

GpuSqlParser::TernaryOperationContext::TernaryOperationContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::TernaryOperationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTernaryOperation(this);
}
void GpuSqlParser::TernaryOperationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTernaryOperation(this);
}
//----------------- AggregationContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::AggregationContext::AGG() {
  return getToken(GpuSqlParser::AGG, 0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::AggregationContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

GpuSqlParser::AggregationContext::AggregationContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::AggregationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAggregation(this);
}
void GpuSqlParser::AggregationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAggregation(this);
}
//----------------- ParenExpressionContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::ParenExpressionContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::ParenExpressionContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}

tree::TerminalNode* GpuSqlParser::ParenExpressionContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

GpuSqlParser::ParenExpressionContext::ParenExpressionContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::ParenExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParenExpression(this);
}
void GpuSqlParser::ParenExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParenExpression(this);
}
//----------------- BinaryOperationContext ------------------------------------------------------------------

std::vector<GpuSqlParser::ExpressionContext *> GpuSqlParser::BinaryOperationContext::expression() {
  return getRuleContexts<GpuSqlParser::ExpressionContext>();
}

GpuSqlParser::ExpressionContext* GpuSqlParser::BinaryOperationContext::expression(size_t i) {
  return getRuleContext<GpuSqlParser::ExpressionContext>(i);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::DIVISION() {
  return getToken(GpuSqlParser::DIVISION, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::ASTERISK() {
  return getToken(GpuSqlParser::ASTERISK, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::PLUS() {
  return getToken(GpuSqlParser::PLUS, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::MINUS() {
  return getToken(GpuSqlParser::MINUS, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GREATER() {
  return getToken(GpuSqlParser::GREATER, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::LESS() {
  return getToken(GpuSqlParser::LESS, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GREATEREQ() {
  return getToken(GpuSqlParser::GREATEREQ, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::LESSEQ() {
  return getToken(GpuSqlParser::LESSEQ, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::EQUALS() {
  return getToken(GpuSqlParser::EQUALS, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::NOTEQUALS() {
  return getToken(GpuSqlParser::NOTEQUALS, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::MODULO() {
  return getToken(GpuSqlParser::MODULO, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO() {
  return getToken(GpuSqlParser::GEO, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::AND() {
  return getToken(GpuSqlParser::AND, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::OR() {
  return getToken(GpuSqlParser::OR, 0);
}

GpuSqlParser::BinaryOperationContext::BinaryOperationContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::BinaryOperationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinaryOperation(this);
}
void GpuSqlParser::BinaryOperationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinaryOperation(this);
}
//----------------- UnaryOperationContext ------------------------------------------------------------------

GpuSqlParser::ExpressionContext* GpuSqlParser::UnaryOperationContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::NOT() {
  return getToken(GpuSqlParser::NOT, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::MINUS() {
  return getToken(GpuSqlParser::MINUS, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::YEAR() {
  return getToken(GpuSqlParser::YEAR, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::MONTH() {
  return getToken(GpuSqlParser::MONTH, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::DAY() {
  return getToken(GpuSqlParser::DAY, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::HOUR() {
  return getToken(GpuSqlParser::HOUR, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::MINUTE() {
  return getToken(GpuSqlParser::MINUTE, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::SECOND() {
  return getToken(GpuSqlParser::SECOND, 0);
}

GpuSqlParser::UnaryOperationContext::UnaryOperationContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::UnaryOperationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryOperation(this);
}
void GpuSqlParser::UnaryOperationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryOperation(this);
}
//----------------- BooleanLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::BooleanLiteralContext::BOOLEANLIT() {
  return getToken(GpuSqlParser::BOOLEANLIT, 0);
}

GpuSqlParser::BooleanLiteralContext::BooleanLiteralContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::BooleanLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBooleanLiteral(this);
}
void GpuSqlParser::BooleanLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBooleanLiteral(this);
}
//----------------- VarReferenceContext ------------------------------------------------------------------

GpuSqlParser::ColumnIdContext* GpuSqlParser::VarReferenceContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
}

GpuSqlParser::VarReferenceContext::VarReferenceContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::VarReferenceContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVarReference(this);
}
void GpuSqlParser::VarReferenceContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVarReference(this);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::expression() {
   return expression(0);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  GpuSqlParser::ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, parentState);
  GpuSqlParser::ExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 64;
  enterRecursionRule(_localctx, 64, GpuSqlParser::RuleExpression, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(333);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::NOT: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;

        setState(283);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOT);
        setState(284);
        expression(27);
        break;
      }

      case GpuSqlParser::MINUS: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(285);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUS);
        setState(286);
        expression(26);
        break;
      }

      case GpuSqlParser::YEAR: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(287);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::YEAR);
        setState(288);
        match(GpuSqlParser::LPAREN);
        setState(289);
        expression(0);
        setState(290);
        match(GpuSqlParser::RPAREN);
        break;
      }

      case GpuSqlParser::MONTH: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(292);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MONTH);
        setState(293);
        match(GpuSqlParser::LPAREN);
        setState(294);
        expression(0);
        setState(295);
        match(GpuSqlParser::RPAREN);
        break;
      }

      case GpuSqlParser::DAY: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(297);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DAY);
        setState(298);
        match(GpuSqlParser::LPAREN);
        setState(299);
        expression(0);
        setState(300);
        match(GpuSqlParser::RPAREN);
        break;
      }

      case GpuSqlParser::HOUR: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(302);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::HOUR);
        setState(303);
        match(GpuSqlParser::LPAREN);
        setState(304);
        expression(0);
        setState(305);
        match(GpuSqlParser::RPAREN);
        break;
      }

      case GpuSqlParser::MINUTE: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(307);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUTE);
        setState(308);
        match(GpuSqlParser::LPAREN);
        setState(309);
        expression(0);
        setState(310);
        match(GpuSqlParser::RPAREN);
        break;
      }

      case GpuSqlParser::SECOND: {
        _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(312);
        dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SECOND);
        setState(313);
        match(GpuSqlParser::LPAREN);
        setState(314);
        expression(0);
        setState(315);
        match(GpuSqlParser::RPAREN);
        break;
      }

      case GpuSqlParser::LPAREN: {
        _localctx = _tracker.createInstance<ParenExpressionContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(317);
        match(GpuSqlParser::LPAREN);
        setState(318);
        expression(0);
        setState(319);
        match(GpuSqlParser::RPAREN);
        break;
      }

      case GpuSqlParser::ID: {
        _localctx = _tracker.createInstance<VarReferenceContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(321);
        columnId();
        break;
      }

      case GpuSqlParser::POINT:
      case GpuSqlParser::MULTIPOINT:
      case GpuSqlParser::LINESTRING:
      case GpuSqlParser::MULTILINESTRING:
      case GpuSqlParser::POLYGON:
      case GpuSqlParser::MULTIPOLYGON: {
        _localctx = _tracker.createInstance<GeoReferenceContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(322);
        geometry();
        break;
      }

      case GpuSqlParser::DATETIMELIT: {
        _localctx = _tracker.createInstance<DateTimeLiteralContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(323);
        match(GpuSqlParser::DATETIMELIT);
        break;
      }

      case GpuSqlParser::FLOATLIT: {
        _localctx = _tracker.createInstance<DecimalLiteralContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(324);
        match(GpuSqlParser::FLOATLIT);
        break;
      }

      case GpuSqlParser::INTLIT: {
        _localctx = _tracker.createInstance<IntLiteralContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(325);
        match(GpuSqlParser::INTLIT);
        break;
      }

      case GpuSqlParser::STRINGLIT: {
        _localctx = _tracker.createInstance<StringLiteralContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(326);
        match(GpuSqlParser::STRINGLIT);
        break;
      }

      case GpuSqlParser::BOOLEANLIT: {
        _localctx = _tracker.createInstance<BooleanLiteralContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(327);
        match(GpuSqlParser::BOOLEANLIT);
        break;
      }

      case GpuSqlParser::AGG: {
        _localctx = _tracker.createInstance<AggregationContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(328);
        match(GpuSqlParser::AGG);
        setState(329);
        match(GpuSqlParser::LPAREN);
        setState(330);
        expression(0);
        setState(331);
        match(GpuSqlParser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(370);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(368);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(335);

          if (!(precpred(_ctx, 19))) throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(336);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == GpuSqlParser::ASTERISK

          || _la == GpuSqlParser::DIVISION)) {
            dynamic_cast<BinaryOperationContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(337);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(20);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(338);

          if (!(precpred(_ctx, 18))) throw FailedPredicateException(this, "precpred(_ctx, 18)");
          setState(339);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == GpuSqlParser::PLUS

          || _la == GpuSqlParser::MINUS)) {
            dynamic_cast<BinaryOperationContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(340);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(19);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(341);

          if (!(precpred(_ctx, 17))) throw FailedPredicateException(this, "precpred(_ctx, 17)");
          setState(342);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == GpuSqlParser::GREATER

          || _la == GpuSqlParser::LESS)) {
            dynamic_cast<BinaryOperationContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(343);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(18);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(344);

          if (!(precpred(_ctx, 16))) throw FailedPredicateException(this, "precpred(_ctx, 16)");
          setState(345);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == GpuSqlParser::GREATEREQ

          || _la == GpuSqlParser::LESSEQ)) {
            dynamic_cast<BinaryOperationContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(346);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(17);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(347);

          if (!(precpred(_ctx, 15))) throw FailedPredicateException(this, "precpred(_ctx, 15)");
          setState(348);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == GpuSqlParser::EQUALS

          || _la == GpuSqlParser::NOTEQUALS)) {
            dynamic_cast<BinaryOperationContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(349);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(16);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(350);

          if (!(precpred(_ctx, 14))) throw FailedPredicateException(this, "precpred(_ctx, 14)");
          setState(351);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MODULO);
          setState(352);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(15);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(353);

          if (!(precpred(_ctx, 13))) throw FailedPredicateException(this, "precpred(_ctx, 13)");
          setState(354);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO);
          setState(355);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(14);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<TernaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(356);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(357);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op = match(GpuSqlParser::BETWEEN);
          setState(358);
          expression(0);
          setState(359);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op2 = match(GpuSqlParser::AND);
          setState(360);
          expression(13);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(362);

          if (!(precpred(_ctx, 11))) throw FailedPredicateException(this, "precpred(_ctx, 11)");
          setState(363);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::AND);
          setState(364);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(12);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(365);

          if (!(precpred(_ctx, 10))) throw FailedPredicateException(this, "precpred(_ctx, 10)");
          setState(366);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::OR);
          setState(367);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(11);
          break;
        }

        } 
      }
      setState(372);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- GeometryContext ------------------------------------------------------------------

GpuSqlParser::GeometryContext::GeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::PointGeometryContext* GpuSqlParser::GeometryContext::pointGeometry() {
  return getRuleContext<GpuSqlParser::PointGeometryContext>(0);
}

GpuSqlParser::PolygonGeometryContext* GpuSqlParser::GeometryContext::polygonGeometry() {
  return getRuleContext<GpuSqlParser::PolygonGeometryContext>(0);
}

GpuSqlParser::LineStringGeometryContext* GpuSqlParser::GeometryContext::lineStringGeometry() {
  return getRuleContext<GpuSqlParser::LineStringGeometryContext>(0);
}

GpuSqlParser::MultiPointGeometryContext* GpuSqlParser::GeometryContext::multiPointGeometry() {
  return getRuleContext<GpuSqlParser::MultiPointGeometryContext>(0);
}

GpuSqlParser::MultiLineStringGeometryContext* GpuSqlParser::GeometryContext::multiLineStringGeometry() {
  return getRuleContext<GpuSqlParser::MultiLineStringGeometryContext>(0);
}

GpuSqlParser::MultiPolygonGeometryContext* GpuSqlParser::GeometryContext::multiPolygonGeometry() {
  return getRuleContext<GpuSqlParser::MultiPolygonGeometryContext>(0);
}


size_t GpuSqlParser::GeometryContext::getRuleIndex() const {
  return GpuSqlParser::RuleGeometry;
}

void GpuSqlParser::GeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGeometry(this);
}

void GpuSqlParser::GeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGeometry(this);
}

GpuSqlParser::GeometryContext* GpuSqlParser::geometry() {
  GeometryContext *_localctx = _tracker.createInstance<GeometryContext>(_ctx, getState());
  enterRule(_localctx, 66, GpuSqlParser::RuleGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(379);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::POINT: {
        setState(373);
        pointGeometry();
        break;
      }

      case GpuSqlParser::POLYGON: {
        setState(374);
        polygonGeometry();
        break;
      }

      case GpuSqlParser::LINESTRING: {
        setState(375);
        lineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOINT: {
        setState(376);
        multiPointGeometry();
        break;
      }

      case GpuSqlParser::MULTILINESTRING: {
        setState(377);
        multiLineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOLYGON: {
        setState(378);
        multiPolygonGeometry();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointGeometryContext ------------------------------------------------------------------

GpuSqlParser::PointGeometryContext::PointGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::PointGeometryContext::POINT() {
  return getToken(GpuSqlParser::POINT, 0);
}

tree::TerminalNode* GpuSqlParser::PointGeometryContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::PointContext* GpuSqlParser::PointGeometryContext::point() {
  return getRuleContext<GpuSqlParser::PointContext>(0);
}

tree::TerminalNode* GpuSqlParser::PointGeometryContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}


size_t GpuSqlParser::PointGeometryContext::getRuleIndex() const {
  return GpuSqlParser::RulePointGeometry;
}

void GpuSqlParser::PointGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPointGeometry(this);
}

void GpuSqlParser::PointGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPointGeometry(this);
}

GpuSqlParser::PointGeometryContext* GpuSqlParser::pointGeometry() {
  PointGeometryContext *_localctx = _tracker.createInstance<PointGeometryContext>(_ctx, getState());
  enterRule(_localctx, 68, GpuSqlParser::RulePointGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(381);
    match(GpuSqlParser::POINT);
    setState(382);
    match(GpuSqlParser::LPAREN);
    setState(383);
    point();
    setState(384);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LineStringGeometryContext ------------------------------------------------------------------

GpuSqlParser::LineStringGeometryContext::LineStringGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::LineStringGeometryContext::LINESTRING() {
  return getToken(GpuSqlParser::LINESTRING, 0);
}

GpuSqlParser::LineStringContext* GpuSqlParser::LineStringGeometryContext::lineString() {
  return getRuleContext<GpuSqlParser::LineStringContext>(0);
}


size_t GpuSqlParser::LineStringGeometryContext::getRuleIndex() const {
  return GpuSqlParser::RuleLineStringGeometry;
}

void GpuSqlParser::LineStringGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLineStringGeometry(this);
}

void GpuSqlParser::LineStringGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLineStringGeometry(this);
}

GpuSqlParser::LineStringGeometryContext* GpuSqlParser::lineStringGeometry() {
  LineStringGeometryContext *_localctx = _tracker.createInstance<LineStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 70, GpuSqlParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(386);
    match(GpuSqlParser::LINESTRING);
    setState(387);
    lineString();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PolygonGeometryContext ------------------------------------------------------------------

GpuSqlParser::PolygonGeometryContext::PolygonGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::PolygonGeometryContext::POLYGON() {
  return getToken(GpuSqlParser::POLYGON, 0);
}

GpuSqlParser::PolygonContext* GpuSqlParser::PolygonGeometryContext::polygon() {
  return getRuleContext<GpuSqlParser::PolygonContext>(0);
}


size_t GpuSqlParser::PolygonGeometryContext::getRuleIndex() const {
  return GpuSqlParser::RulePolygonGeometry;
}

void GpuSqlParser::PolygonGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPolygonGeometry(this);
}

void GpuSqlParser::PolygonGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPolygonGeometry(this);
}

GpuSqlParser::PolygonGeometryContext* GpuSqlParser::polygonGeometry() {
  PolygonGeometryContext *_localctx = _tracker.createInstance<PolygonGeometryContext>(_ctx, getState());
  enterRule(_localctx, 72, GpuSqlParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(389);
    match(GpuSqlParser::POLYGON);
    setState(390);
    polygon();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiPointGeometryContext ------------------------------------------------------------------

GpuSqlParser::MultiPointGeometryContext::MultiPointGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::MultiPointGeometryContext::MULTIPOINT() {
  return getToken(GpuSqlParser::MULTIPOINT, 0);
}

tree::TerminalNode* GpuSqlParser::MultiPointGeometryContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

std::vector<GpuSqlParser::PointOrClosedPointContext *> GpuSqlParser::MultiPointGeometryContext::pointOrClosedPoint() {
  return getRuleContexts<GpuSqlParser::PointOrClosedPointContext>();
}

GpuSqlParser::PointOrClosedPointContext* GpuSqlParser::MultiPointGeometryContext::pointOrClosedPoint(size_t i) {
  return getRuleContext<GpuSqlParser::PointOrClosedPointContext>(i);
}

tree::TerminalNode* GpuSqlParser::MultiPointGeometryContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::MultiPointGeometryContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::MultiPointGeometryContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::MultiPointGeometryContext::getRuleIndex() const {
  return GpuSqlParser::RuleMultiPointGeometry;
}

void GpuSqlParser::MultiPointGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiPointGeometry(this);
}

void GpuSqlParser::MultiPointGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiPointGeometry(this);
}

GpuSqlParser::MultiPointGeometryContext* GpuSqlParser::multiPointGeometry() {
  MultiPointGeometryContext *_localctx = _tracker.createInstance<MultiPointGeometryContext>(_ctx, getState());
  enterRule(_localctx, 74, GpuSqlParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(392);
    match(GpuSqlParser::MULTIPOINT);
    setState(393);
    match(GpuSqlParser::LPAREN);
    setState(394);
    pointOrClosedPoint();
    setState(399);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(395);
      match(GpuSqlParser::COMMA);
      setState(396);
      pointOrClosedPoint();
      setState(401);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(402);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiLineStringGeometryContext ------------------------------------------------------------------

GpuSqlParser::MultiLineStringGeometryContext::MultiLineStringGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::MultiLineStringGeometryContext::MULTILINESTRING() {
  return getToken(GpuSqlParser::MULTILINESTRING, 0);
}

tree::TerminalNode* GpuSqlParser::MultiLineStringGeometryContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

std::vector<GpuSqlParser::LineStringContext *> GpuSqlParser::MultiLineStringGeometryContext::lineString() {
  return getRuleContexts<GpuSqlParser::LineStringContext>();
}

GpuSqlParser::LineStringContext* GpuSqlParser::MultiLineStringGeometryContext::lineString(size_t i) {
  return getRuleContext<GpuSqlParser::LineStringContext>(i);
}

tree::TerminalNode* GpuSqlParser::MultiLineStringGeometryContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::MultiLineStringGeometryContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::MultiLineStringGeometryContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::MultiLineStringGeometryContext::getRuleIndex() const {
  return GpuSqlParser::RuleMultiLineStringGeometry;
}

void GpuSqlParser::MultiLineStringGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiLineStringGeometry(this);
}

void GpuSqlParser::MultiLineStringGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiLineStringGeometry(this);
}

GpuSqlParser::MultiLineStringGeometryContext* GpuSqlParser::multiLineStringGeometry() {
  MultiLineStringGeometryContext *_localctx = _tracker.createInstance<MultiLineStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 76, GpuSqlParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(404);
    match(GpuSqlParser::MULTILINESTRING);
    setState(405);
    match(GpuSqlParser::LPAREN);
    setState(406);
    lineString();
    setState(411);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(407);
      match(GpuSqlParser::COMMA);
      setState(408);
      lineString();
      setState(413);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(414);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiPolygonGeometryContext ------------------------------------------------------------------

GpuSqlParser::MultiPolygonGeometryContext::MultiPolygonGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::MultiPolygonGeometryContext::MULTIPOLYGON() {
  return getToken(GpuSqlParser::MULTIPOLYGON, 0);
}

tree::TerminalNode* GpuSqlParser::MultiPolygonGeometryContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

std::vector<GpuSqlParser::PolygonContext *> GpuSqlParser::MultiPolygonGeometryContext::polygon() {
  return getRuleContexts<GpuSqlParser::PolygonContext>();
}

GpuSqlParser::PolygonContext* GpuSqlParser::MultiPolygonGeometryContext::polygon(size_t i) {
  return getRuleContext<GpuSqlParser::PolygonContext>(i);
}

tree::TerminalNode* GpuSqlParser::MultiPolygonGeometryContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::MultiPolygonGeometryContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::MultiPolygonGeometryContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::MultiPolygonGeometryContext::getRuleIndex() const {
  return GpuSqlParser::RuleMultiPolygonGeometry;
}

void GpuSqlParser::MultiPolygonGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiPolygonGeometry(this);
}

void GpuSqlParser::MultiPolygonGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiPolygonGeometry(this);
}

GpuSqlParser::MultiPolygonGeometryContext* GpuSqlParser::multiPolygonGeometry() {
  MultiPolygonGeometryContext *_localctx = _tracker.createInstance<MultiPolygonGeometryContext>(_ctx, getState());
  enterRule(_localctx, 78, GpuSqlParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(416);
    match(GpuSqlParser::MULTIPOLYGON);
    setState(417);
    match(GpuSqlParser::LPAREN);
    setState(418);
    polygon();
    setState(423);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(419);
      match(GpuSqlParser::COMMA);
      setState(420);
      polygon();
      setState(425);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(426);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointOrClosedPointContext ------------------------------------------------------------------

GpuSqlParser::PointOrClosedPointContext::PointOrClosedPointContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::PointContext* GpuSqlParser::PointOrClosedPointContext::point() {
  return getRuleContext<GpuSqlParser::PointContext>(0);
}

tree::TerminalNode* GpuSqlParser::PointOrClosedPointContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::PointOrClosedPointContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}


size_t GpuSqlParser::PointOrClosedPointContext::getRuleIndex() const {
  return GpuSqlParser::RulePointOrClosedPoint;
}

void GpuSqlParser::PointOrClosedPointContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPointOrClosedPoint(this);
}

void GpuSqlParser::PointOrClosedPointContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPointOrClosedPoint(this);
}

GpuSqlParser::PointOrClosedPointContext* GpuSqlParser::pointOrClosedPoint() {
  PointOrClosedPointContext *_localctx = _tracker.createInstance<PointOrClosedPointContext>(_ctx, getState());
  enterRule(_localctx, 80, GpuSqlParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(433);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT: {
        enterOuterAlt(_localctx, 1);
        setState(428);
        point();
        break;
      }

      case GpuSqlParser::LPAREN: {
        enterOuterAlt(_localctx, 2);
        setState(429);
        match(GpuSqlParser::LPAREN);
        setState(430);
        point();
        setState(431);
        match(GpuSqlParser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PolygonContext ------------------------------------------------------------------

GpuSqlParser::PolygonContext::PolygonContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::PolygonContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

std::vector<GpuSqlParser::LineStringContext *> GpuSqlParser::PolygonContext::lineString() {
  return getRuleContexts<GpuSqlParser::LineStringContext>();
}

GpuSqlParser::LineStringContext* GpuSqlParser::PolygonContext::lineString(size_t i) {
  return getRuleContext<GpuSqlParser::LineStringContext>(i);
}

tree::TerminalNode* GpuSqlParser::PolygonContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::PolygonContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::PolygonContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::PolygonContext::getRuleIndex() const {
  return GpuSqlParser::RulePolygon;
}

void GpuSqlParser::PolygonContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPolygon(this);
}

void GpuSqlParser::PolygonContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPolygon(this);
}

GpuSqlParser::PolygonContext* GpuSqlParser::polygon() {
  PolygonContext *_localctx = _tracker.createInstance<PolygonContext>(_ctx, getState());
  enterRule(_localctx, 82, GpuSqlParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(435);
    match(GpuSqlParser::LPAREN);
    setState(436);
    lineString();
    setState(441);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(437);
      match(GpuSqlParser::COMMA);
      setState(438);
      lineString();
      setState(443);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(444);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LineStringContext ------------------------------------------------------------------

GpuSqlParser::LineStringContext::LineStringContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::LineStringContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

std::vector<GpuSqlParser::PointContext *> GpuSqlParser::LineStringContext::point() {
  return getRuleContexts<GpuSqlParser::PointContext>();
}

GpuSqlParser::PointContext* GpuSqlParser::LineStringContext::point(size_t i) {
  return getRuleContext<GpuSqlParser::PointContext>(i);
}

tree::TerminalNode* GpuSqlParser::LineStringContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::LineStringContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::LineStringContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::LineStringContext::getRuleIndex() const {
  return GpuSqlParser::RuleLineString;
}

void GpuSqlParser::LineStringContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLineString(this);
}

void GpuSqlParser::LineStringContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLineString(this);
}

GpuSqlParser::LineStringContext* GpuSqlParser::lineString() {
  LineStringContext *_localctx = _tracker.createInstance<LineStringContext>(_ctx, getState());
  enterRule(_localctx, 84, GpuSqlParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(446);
    match(GpuSqlParser::LPAREN);
    setState(447);
    point();
    setState(452);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(448);
      match(GpuSqlParser::COMMA);
      setState(449);
      point();
      setState(454);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(455);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointContext ------------------------------------------------------------------

GpuSqlParser::PointContext::PointContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> GpuSqlParser::PointContext::FLOATLIT() {
  return getTokens(GpuSqlParser::FLOATLIT);
}

tree::TerminalNode* GpuSqlParser::PointContext::FLOATLIT(size_t i) {
  return getToken(GpuSqlParser::FLOATLIT, i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::PointContext::INTLIT() {
  return getTokens(GpuSqlParser::INTLIT);
}

tree::TerminalNode* GpuSqlParser::PointContext::INTLIT(size_t i) {
  return getToken(GpuSqlParser::INTLIT, i);
}


size_t GpuSqlParser::PointContext::getRuleIndex() const {
  return GpuSqlParser::RulePoint;
}

void GpuSqlParser::PointContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPoint(this);
}

void GpuSqlParser::PointContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPoint(this);
}

GpuSqlParser::PointContext* GpuSqlParser::point() {
  PointContext *_localctx = _tracker.createInstance<PointContext>(_ctx, getState());
  enterRule(_localctx, 86, GpuSqlParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(457);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FLOATLIT

    || _la == GpuSqlParser::INTLIT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(458);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FLOATLIT

    || _la == GpuSqlParser::INTLIT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool GpuSqlParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 32: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool GpuSqlParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 19);
    case 1: return precpred(_ctx, 18);
    case 2: return precpred(_ctx, 17);
    case 3: return precpred(_ctx, 16);
    case 4: return precpred(_ctx, 15);
    case 5: return precpred(_ctx, 14);
    case 6: return precpred(_ctx, 13);
    case 7: return precpred(_ctx, 12);
    case 8: return precpred(_ctx, 11);
    case 9: return precpred(_ctx, 10);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> GpuSqlParser::_decisionToDFA;
atn::PredictionContextCache GpuSqlParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN GpuSqlParser::_atn;
std::vector<uint16_t> GpuSqlParser::_serializedATN;

std::vector<std::string> GpuSqlParser::_ruleNames = {
  "sqlFile", "statement", "showStatement", "showDatabases", "showTables", 
  "showColumns", "sqlSelect", "sqlCreateDb", "sqlCreateTable", "sqlInsertInto", 
  "newTableColumns", "newTableColumn", "selectColumns", "selectColumn", 
  "whereClause", "orderByColumns", "orderByColumn", "insertIntoValues", 
  "insertIntoColumns", "groupByColumns", "groupByColumn", "columnId", "fromTables", 
  "joinClauses", "joinClause", "joinTable", "table", "column", "database", 
  "limit", "offset", "columnValue", "expression", "geometry", "pointGeometry", 
  "lineStringGeometry", "polygonGeometry", "multiPointGeometry", "multiLineStringGeometry", 
  "multiPolygonGeometry", "pointOrClosedPoint", "polygon", "lineString", 
  "point"
};

std::vector<std::string> GpuSqlParser::_literalNames = {
  "", "", "'\n'", "'\r'", "'\r\n'", "", "';'", "'''", "'\"'", "':'", "','", 
  "'.'", "", "", "", "'POINT'", "'MULTIPOINT'", "'LINESTRING'", "'MULTILINESTRING'", 
  "'POLYGON'", "'MULTIPOLYGON'", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "'+'", "'-'", 
  "'*'", "'/'", "'%'", "'='", "'!='", "'('", "')'", "'>'", "'<'", "'>='", 
  "'<='", "'!'"
};

std::vector<std::string> GpuSqlParser::_symbolicNames = {
  "", "DATETIMELIT", "LF", "CR", "CRLF", "WS", "SEMICOL", "SQOUTE", "DQOUTE", 
  "COLON", "COMMA", "DOT", "DATELIT", "TIMELIT", "DATATYPE", "POINT", "MULTIPOINT", 
  "LINESTRING", "MULTILINESTRING", "POLYGON", "MULTIPOLYGON", "INTTYPE", 
  "LONGTYPE", "FLOATTYPE", "DOUBLETYPE", "STRINGTYPE", "BOOLEANTYPE", "POINTTYPE", 
  "POLYTYPE", "INSERTINTO", "CREATEDB", "CREATETABLE", "VALUES", "SELECT", 
  "FROM", "JOIN", "WHERE", "GROUPBY", "AS", "IN", "BETWEEN", "ON", "ORDERBY", 
  "DIR", "LIMIT", "OFFSET", "SHOWDB", "SHOWTB", "SHOWCL", "AGG", "AVG", 
  "SUM", "MIN", "MAX", "COUNT", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", 
  "SECOND", "GEO", "CONTAINS", "PLUS", "MINUS", "ASTERISK", "DIVISION", 
  "MODULO", "EQUALS", "NOTEQUALS", "LPAREN", "RPAREN", "GREATER", "LESS", 
  "GREATEREQ", "LESSEQ", "NOT", "OR", "AND", "FLOATLIT", "INTLIT", "ID", 
  "BOOLEANLIT", "STRINGLIT"
};

dfa::Vocabulary GpuSqlParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> GpuSqlParser::_tokenNames;

GpuSqlParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x55, 0x1cf, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 
    0x1f, 0x9, 0x1f, 0x4, 0x20, 0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x4, 0x22, 
    0x9, 0x22, 0x4, 0x23, 0x9, 0x23, 0x4, 0x24, 0x9, 0x24, 0x4, 0x25, 0x9, 
    0x25, 0x4, 0x26, 0x9, 0x26, 0x4, 0x27, 0x9, 0x27, 0x4, 0x28, 0x9, 0x28, 
    0x4, 0x29, 0x9, 0x29, 0x4, 0x2a, 0x9, 0x2a, 0x4, 0x2b, 0x9, 0x2b, 0x4, 
    0x2c, 0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x3, 0x2, 0x7, 0x2, 0x5c, 0xa, 
    0x2, 0xc, 0x2, 0xe, 0x2, 0x5f, 0xb, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x68, 0xa, 0x3, 0x3, 
    0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 0x6d, 0xa, 0x4, 0x3, 0x5, 0x3, 0x5, 
    0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x75, 0xa, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 
    0x7, 0x7e, 0xa, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x87, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 
    0x8, 0x8b, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x8f, 0xa, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x5, 0x8, 0x93, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 
    0x97, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x9b, 0xa, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x7, 0xc, 0xb7, 0xa, 0xc, 0xc, 
    0xc, 0xe, 0xc, 0xba, 0xb, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x7, 0xe, 0xc2, 0xa, 0xe, 0xc, 0xe, 0xe, 0xe, 0xc5, 
    0xb, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x11, 0x7, 0x11, 0xce, 0xa, 0x11, 0xc, 0x11, 0xe, 0x11, 0xd1, 
    0xb, 0x11, 0x3, 0x12, 0x3, 0x12, 0x5, 0x12, 0xd5, 0xa, 0x12, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x13, 0x7, 0x13, 0xda, 0xa, 0x13, 0xc, 0x13, 0xe, 0x13, 
    0xdd, 0xb, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x7, 0x14, 0xe2, 0xa, 
    0x14, 0xc, 0x14, 0xe, 0x14, 0xe5, 0xb, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x15, 0x7, 0x15, 0xea, 0xa, 0x15, 0xc, 0x15, 0xe, 0x15, 0xed, 0xb, 0x15, 
    0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 
    0x17, 0x5, 0x17, 0xf6, 0xa, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x7, 
    0x18, 0xfb, 0xa, 0x18, 0xc, 0x18, 0xe, 0x18, 0xfe, 0xb, 0x18, 0x3, 0x19, 
    0x6, 0x19, 0x101, 0xa, 0x19, 0xd, 0x19, 0xe, 0x19, 0x102, 0x3, 0x1a, 
    0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 
    0x1c, 0x3, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1f, 
    0x3, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 
    0x21, 0x3, 0x21, 0x5, 0x21, 0x11b, 0xa, 0x21, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 0x150, 0xa, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x7, 0x22, 0x173, 0xa, 0x22, 0xc, 0x22, 0xe, 0x22, 0x176, 0xb, 
    0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x5, 0x23, 0x17e, 0xa, 0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 
    0x3, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 
    0x26, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 
    0x190, 0xa, 0x27, 0xc, 0x27, 0xe, 0x27, 0x193, 0xb, 0x27, 0x3, 0x27, 
    0x3, 0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x7, 
    0x28, 0x19c, 0xa, 0x28, 0xc, 0x28, 0xe, 0x28, 0x19f, 0xb, 0x28, 0x3, 
    0x28, 0x3, 0x28, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 
    0x7, 0x29, 0x1a8, 0xa, 0x29, 0xc, 0x29, 0xe, 0x29, 0x1ab, 0xb, 0x29, 
    0x3, 0x29, 0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 
    0x2a, 0x5, 0x2a, 0x1b4, 0xa, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 
    0x3, 0x2b, 0x7, 0x2b, 0x1ba, 0xa, 0x2b, 0xc, 0x2b, 0xe, 0x2b, 0x1bd, 
    0xb, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 
    0x2c, 0x7, 0x2c, 0x1c5, 0xa, 0x2c, 0xc, 0x2c, 0xe, 0x2c, 0x1c8, 0xb, 
    0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 
    0x2, 0x3, 0x42, 0x2e, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 
    0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 
    0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 
    0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x2, 
    0x9, 0x4, 0x2, 0x24, 0x24, 0x29, 0x29, 0x3, 0x2, 0x43, 0x44, 0x3, 0x2, 
    0x41, 0x42, 0x3, 0x2, 0x4a, 0x4b, 0x3, 0x2, 0x4c, 0x4d, 0x3, 0x2, 0x46, 
    0x47, 0x3, 0x2, 0x51, 0x52, 0x2, 0x1e4, 0x2, 0x5d, 0x3, 0x2, 0x2, 0x2, 
    0x4, 0x67, 0x3, 0x2, 0x2, 0x2, 0x6, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x8, 0x6e, 
    0x3, 0x2, 0x2, 0x2, 0xa, 0x71, 0x3, 0x2, 0x2, 0x2, 0xc, 0x78, 0x3, 0x2, 
    0x2, 0x2, 0xe, 0x81, 0x3, 0x2, 0x2, 0x2, 0x10, 0x9e, 0x3, 0x2, 0x2, 
    0x2, 0x12, 0xa2, 0x3, 0x2, 0x2, 0x2, 0x14, 0xa8, 0x3, 0x2, 0x2, 0x2, 
    0x16, 0xb3, 0x3, 0x2, 0x2, 0x2, 0x18, 0xbb, 0x3, 0x2, 0x2, 0x2, 0x1a, 
    0xbe, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xc6, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xc8, 
    0x3, 0x2, 0x2, 0x2, 0x20, 0xca, 0x3, 0x2, 0x2, 0x2, 0x22, 0xd2, 0x3, 
    0x2, 0x2, 0x2, 0x24, 0xd6, 0x3, 0x2, 0x2, 0x2, 0x26, 0xde, 0x3, 0x2, 
    0x2, 0x2, 0x28, 0xe6, 0x3, 0x2, 0x2, 0x2, 0x2a, 0xee, 0x3, 0x2, 0x2, 
    0x2, 0x2c, 0xf5, 0x3, 0x2, 0x2, 0x2, 0x2e, 0xf7, 0x3, 0x2, 0x2, 0x2, 
    0x30, 0x100, 0x3, 0x2, 0x2, 0x2, 0x32, 0x104, 0x3, 0x2, 0x2, 0x2, 0x34, 
    0x109, 0x3, 0x2, 0x2, 0x2, 0x36, 0x10b, 0x3, 0x2, 0x2, 0x2, 0x38, 0x10d, 
    0x3, 0x2, 0x2, 0x2, 0x3a, 0x10f, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x111, 0x3, 
    0x2, 0x2, 0x2, 0x3e, 0x113, 0x3, 0x2, 0x2, 0x2, 0x40, 0x11a, 0x3, 0x2, 
    0x2, 0x2, 0x42, 0x14f, 0x3, 0x2, 0x2, 0x2, 0x44, 0x17d, 0x3, 0x2, 0x2, 
    0x2, 0x46, 0x17f, 0x3, 0x2, 0x2, 0x2, 0x48, 0x184, 0x3, 0x2, 0x2, 0x2, 
    0x4a, 0x187, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x18a, 0x3, 0x2, 0x2, 0x2, 0x4e, 
    0x196, 0x3, 0x2, 0x2, 0x2, 0x50, 0x1a2, 0x3, 0x2, 0x2, 0x2, 0x52, 0x1b3, 
    0x3, 0x2, 0x2, 0x2, 0x54, 0x1b5, 0x3, 0x2, 0x2, 0x2, 0x56, 0x1c0, 0x3, 
    0x2, 0x2, 0x2, 0x58, 0x1cb, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x5c, 0x5, 0x4, 
    0x3, 0x2, 0x5b, 0x5a, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x5f, 0x3, 0x2, 0x2, 
    0x2, 0x5d, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x5d, 0x5e, 0x3, 0x2, 0x2, 0x2, 
    0x5e, 0x60, 0x3, 0x2, 0x2, 0x2, 0x5f, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x60, 
    0x61, 0x7, 0x2, 0x2, 0x3, 0x61, 0x3, 0x3, 0x2, 0x2, 0x2, 0x62, 0x68, 
    0x5, 0xe, 0x8, 0x2, 0x63, 0x68, 0x5, 0x10, 0x9, 0x2, 0x64, 0x68, 0x5, 
    0x12, 0xa, 0x2, 0x65, 0x68, 0x5, 0x14, 0xb, 0x2, 0x66, 0x68, 0x5, 0x6, 
    0x4, 0x2, 0x67, 0x62, 0x3, 0x2, 0x2, 0x2, 0x67, 0x63, 0x3, 0x2, 0x2, 
    0x2, 0x67, 0x64, 0x3, 0x2, 0x2, 0x2, 0x67, 0x65, 0x3, 0x2, 0x2, 0x2, 
    0x67, 0x66, 0x3, 0x2, 0x2, 0x2, 0x68, 0x5, 0x3, 0x2, 0x2, 0x2, 0x69, 
    0x6d, 0x5, 0x8, 0x5, 0x2, 0x6a, 0x6d, 0x5, 0xa, 0x6, 0x2, 0x6b, 0x6d, 
    0x5, 0xc, 0x7, 0x2, 0x6c, 0x69, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x6a, 0x3, 
    0x2, 0x2, 0x2, 0x6c, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x6d, 0x7, 0x3, 0x2, 
    0x2, 0x2, 0x6e, 0x6f, 0x7, 0x30, 0x2, 0x2, 0x6f, 0x70, 0x7, 0x8, 0x2, 
    0x2, 0x70, 0x9, 0x3, 0x2, 0x2, 0x2, 0x71, 0x74, 0x7, 0x31, 0x2, 0x2, 
    0x72, 0x73, 0x9, 0x2, 0x2, 0x2, 0x73, 0x75, 0x5, 0x3a, 0x1e, 0x2, 0x74, 
    0x72, 0x3, 0x2, 0x2, 0x2, 0x74, 0x75, 0x3, 0x2, 0x2, 0x2, 0x75, 0x76, 
    0x3, 0x2, 0x2, 0x2, 0x76, 0x77, 0x7, 0x8, 0x2, 0x2, 0x77, 0xb, 0x3, 
    0x2, 0x2, 0x2, 0x78, 0x79, 0x7, 0x32, 0x2, 0x2, 0x79, 0x7a, 0x9, 0x2, 
    0x2, 0x2, 0x7a, 0x7d, 0x5, 0x36, 0x1c, 0x2, 0x7b, 0x7c, 0x9, 0x2, 0x2, 
    0x2, 0x7c, 0x7e, 0x5, 0x3a, 0x1e, 0x2, 0x7d, 0x7b, 0x3, 0x2, 0x2, 0x2, 
    0x7d, 0x7e, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x7f, 
    0x80, 0x7, 0x8, 0x2, 0x2, 0x80, 0xd, 0x3, 0x2, 0x2, 0x2, 0x81, 0x82, 
    0x7, 0x23, 0x2, 0x2, 0x82, 0x83, 0x5, 0x1a, 0xe, 0x2, 0x83, 0x84, 0x7, 
    0x24, 0x2, 0x2, 0x84, 0x86, 0x5, 0x2e, 0x18, 0x2, 0x85, 0x87, 0x5, 0x30, 
    0x19, 0x2, 0x86, 0x85, 0x3, 0x2, 0x2, 0x2, 0x86, 0x87, 0x3, 0x2, 0x2, 
    0x2, 0x87, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x88, 0x89, 0x7, 0x26, 0x2, 0x2, 
    0x89, 0x8b, 0x5, 0x1e, 0x10, 0x2, 0x8a, 0x88, 0x3, 0x2, 0x2, 0x2, 0x8a, 
    0x8b, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x8d, 
    0x7, 0x27, 0x2, 0x2, 0x8d, 0x8f, 0x5, 0x28, 0x15, 0x2, 0x8e, 0x8c, 0x3, 
    0x2, 0x2, 0x2, 0x8e, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x8f, 0x92, 0x3, 0x2, 
    0x2, 0x2, 0x90, 0x91, 0x7, 0x2c, 0x2, 0x2, 0x91, 0x93, 0x5, 0x20, 0x11, 
    0x2, 0x92, 0x90, 0x3, 0x2, 0x2, 0x2, 0x92, 0x93, 0x3, 0x2, 0x2, 0x2, 
    0x93, 0x96, 0x3, 0x2, 0x2, 0x2, 0x94, 0x95, 0x7, 0x2e, 0x2, 0x2, 0x95, 
    0x97, 0x5, 0x3c, 0x1f, 0x2, 0x96, 0x94, 0x3, 0x2, 0x2, 0x2, 0x96, 0x97, 
    0x3, 0x2, 0x2, 0x2, 0x97, 0x9a, 0x3, 0x2, 0x2, 0x2, 0x98, 0x99, 0x7, 
    0x2f, 0x2, 0x2, 0x99, 0x9b, 0x5, 0x3e, 0x20, 0x2, 0x9a, 0x98, 0x3, 0x2, 
    0x2, 0x2, 0x9a, 0x9b, 0x3, 0x2, 0x2, 0x2, 0x9b, 0x9c, 0x3, 0x2, 0x2, 
    0x2, 0x9c, 0x9d, 0x7, 0x8, 0x2, 0x2, 0x9d, 0xf, 0x3, 0x2, 0x2, 0x2, 
    0x9e, 0x9f, 0x7, 0x20, 0x2, 0x2, 0x9f, 0xa0, 0x5, 0x3a, 0x1e, 0x2, 0xa0, 
    0xa1, 0x7, 0x8, 0x2, 0x2, 0xa1, 0x11, 0x3, 0x2, 0x2, 0x2, 0xa2, 0xa3, 
    0x7, 0x21, 0x2, 0x2, 0xa3, 0xa4, 0x5, 0x36, 0x1c, 0x2, 0xa4, 0xa5, 0x7, 
    0x48, 0x2, 0x2, 0xa5, 0xa6, 0x5, 0x16, 0xc, 0x2, 0xa6, 0xa7, 0x7, 0x49, 
    0x2, 0x2, 0xa7, 0x13, 0x3, 0x2, 0x2, 0x2, 0xa8, 0xa9, 0x7, 0x1f, 0x2, 
    0x2, 0xa9, 0xaa, 0x5, 0x36, 0x1c, 0x2, 0xaa, 0xab, 0x7, 0x48, 0x2, 0x2, 
    0xab, 0xac, 0x5, 0x26, 0x14, 0x2, 0xac, 0xad, 0x7, 0x49, 0x2, 0x2, 0xad, 
    0xae, 0x7, 0x22, 0x2, 0x2, 0xae, 0xaf, 0x7, 0x48, 0x2, 0x2, 0xaf, 0xb0, 
    0x5, 0x24, 0x13, 0x2, 0xb0, 0xb1, 0x7, 0x49, 0x2, 0x2, 0xb1, 0xb2, 0x7, 
    0x8, 0x2, 0x2, 0xb2, 0x15, 0x3, 0x2, 0x2, 0x2, 0xb3, 0xb8, 0x5, 0x18, 
    0xd, 0x2, 0xb4, 0xb5, 0x7, 0xc, 0x2, 0x2, 0xb5, 0xb7, 0x5, 0x18, 0xd, 
    0x2, 0xb6, 0xb4, 0x3, 0x2, 0x2, 0x2, 0xb7, 0xba, 0x3, 0x2, 0x2, 0x2, 
    0xb8, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xb9, 0x3, 0x2, 0x2, 0x2, 0xb9, 
    0x17, 0x3, 0x2, 0x2, 0x2, 0xba, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xbc, 
    0x5, 0x2c, 0x17, 0x2, 0xbc, 0xbd, 0x7, 0x10, 0x2, 0x2, 0xbd, 0x19, 0x3, 
    0x2, 0x2, 0x2, 0xbe, 0xc3, 0x5, 0x1c, 0xf, 0x2, 0xbf, 0xc0, 0x7, 0xc, 
    0x2, 0x2, 0xc0, 0xc2, 0x5, 0x1c, 0xf, 0x2, 0xc1, 0xbf, 0x3, 0x2, 0x2, 
    0x2, 0xc2, 0xc5, 0x3, 0x2, 0x2, 0x2, 0xc3, 0xc1, 0x3, 0x2, 0x2, 0x2, 
    0xc3, 0xc4, 0x3, 0x2, 0x2, 0x2, 0xc4, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xc5, 
    0xc3, 0x3, 0x2, 0x2, 0x2, 0xc6, 0xc7, 0x5, 0x42, 0x22, 0x2, 0xc7, 0x1d, 
    0x3, 0x2, 0x2, 0x2, 0xc8, 0xc9, 0x5, 0x42, 0x22, 0x2, 0xc9, 0x1f, 0x3, 
    0x2, 0x2, 0x2, 0xca, 0xcf, 0x5, 0x22, 0x12, 0x2, 0xcb, 0xcc, 0x7, 0xc, 
    0x2, 0x2, 0xcc, 0xce, 0x5, 0x22, 0x12, 0x2, 0xcd, 0xcb, 0x3, 0x2, 0x2, 
    0x2, 0xce, 0xd1, 0x3, 0x2, 0x2, 0x2, 0xcf, 0xcd, 0x3, 0x2, 0x2, 0x2, 
    0xcf, 0xd0, 0x3, 0x2, 0x2, 0x2, 0xd0, 0x21, 0x3, 0x2, 0x2, 0x2, 0xd1, 
    0xcf, 0x3, 0x2, 0x2, 0x2, 0xd2, 0xd4, 0x5, 0x2c, 0x17, 0x2, 0xd3, 0xd5, 
    0x7, 0x2d, 0x2, 0x2, 0xd4, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd5, 0x3, 
    0x2, 0x2, 0x2, 0xd5, 0x23, 0x3, 0x2, 0x2, 0x2, 0xd6, 0xdb, 0x5, 0x40, 
    0x21, 0x2, 0xd7, 0xd8, 0x7, 0xc, 0x2, 0x2, 0xd8, 0xda, 0x5, 0x40, 0x21, 
    0x2, 0xd9, 0xd7, 0x3, 0x2, 0x2, 0x2, 0xda, 0xdd, 0x3, 0x2, 0x2, 0x2, 
    0xdb, 0xd9, 0x3, 0x2, 0x2, 0x2, 0xdb, 0xdc, 0x3, 0x2, 0x2, 0x2, 0xdc, 
    0x25, 0x3, 0x2, 0x2, 0x2, 0xdd, 0xdb, 0x3, 0x2, 0x2, 0x2, 0xde, 0xe3, 
    0x5, 0x2c, 0x17, 0x2, 0xdf, 0xe0, 0x7, 0xc, 0x2, 0x2, 0xe0, 0xe2, 0x5, 
    0x2c, 0x17, 0x2, 0xe1, 0xdf, 0x3, 0x2, 0x2, 0x2, 0xe2, 0xe5, 0x3, 0x2, 
    0x2, 0x2, 0xe3, 0xe1, 0x3, 0x2, 0x2, 0x2, 0xe3, 0xe4, 0x3, 0x2, 0x2, 
    0x2, 0xe4, 0x27, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe3, 0x3, 0x2, 0x2, 0x2, 
    0xe6, 0xeb, 0x5, 0x2a, 0x16, 0x2, 0xe7, 0xe8, 0x7, 0xc, 0x2, 0x2, 0xe8, 
    0xea, 0x5, 0x2a, 0x16, 0x2, 0xe9, 0xe7, 0x3, 0x2, 0x2, 0x2, 0xea, 0xed, 
    0x3, 0x2, 0x2, 0x2, 0xeb, 0xe9, 0x3, 0x2, 0x2, 0x2, 0xeb, 0xec, 0x3, 
    0x2, 0x2, 0x2, 0xec, 0x29, 0x3, 0x2, 0x2, 0x2, 0xed, 0xeb, 0x3, 0x2, 
    0x2, 0x2, 0xee, 0xef, 0x5, 0x42, 0x22, 0x2, 0xef, 0x2b, 0x3, 0x2, 0x2, 
    0x2, 0xf0, 0xf6, 0x5, 0x38, 0x1d, 0x2, 0xf1, 0xf2, 0x5, 0x36, 0x1c, 
    0x2, 0xf2, 0xf3, 0x7, 0xd, 0x2, 0x2, 0xf3, 0xf4, 0x5, 0x38, 0x1d, 0x2, 
    0xf4, 0xf6, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xf0, 0x3, 0x2, 0x2, 0x2, 0xf5, 
    0xf1, 0x3, 0x2, 0x2, 0x2, 0xf6, 0x2d, 0x3, 0x2, 0x2, 0x2, 0xf7, 0xfc, 
    0x5, 0x36, 0x1c, 0x2, 0xf8, 0xf9, 0x7, 0xc, 0x2, 0x2, 0xf9, 0xfb, 0x5, 
    0x36, 0x1c, 0x2, 0xfa, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfe, 0x3, 0x2, 
    0x2, 0x2, 0xfc, 0xfa, 0x3, 0x2, 0x2, 0x2, 0xfc, 0xfd, 0x3, 0x2, 0x2, 
    0x2, 0xfd, 0x2f, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xfc, 0x3, 0x2, 0x2, 0x2, 
    0xff, 0x101, 0x5, 0x32, 0x1a, 0x2, 0x100, 0xff, 0x3, 0x2, 0x2, 0x2, 
    0x101, 0x102, 0x3, 0x2, 0x2, 0x2, 0x102, 0x100, 0x3, 0x2, 0x2, 0x2, 
    0x102, 0x103, 0x3, 0x2, 0x2, 0x2, 0x103, 0x31, 0x3, 0x2, 0x2, 0x2, 0x104, 
    0x105, 0x7, 0x25, 0x2, 0x2, 0x105, 0x106, 0x5, 0x34, 0x1b, 0x2, 0x106, 
    0x107, 0x7, 0x2b, 0x2, 0x2, 0x107, 0x108, 0x5, 0x42, 0x22, 0x2, 0x108, 
    0x33, 0x3, 0x2, 0x2, 0x2, 0x109, 0x10a, 0x5, 0x36, 0x1c, 0x2, 0x10a, 
    0x35, 0x3, 0x2, 0x2, 0x2, 0x10b, 0x10c, 0x7, 0x53, 0x2, 0x2, 0x10c, 
    0x37, 0x3, 0x2, 0x2, 0x2, 0x10d, 0x10e, 0x7, 0x53, 0x2, 0x2, 0x10e, 
    0x39, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x110, 0x7, 0x53, 0x2, 0x2, 0x110, 
    0x3b, 0x3, 0x2, 0x2, 0x2, 0x111, 0x112, 0x7, 0x52, 0x2, 0x2, 0x112, 
    0x3d, 0x3, 0x2, 0x2, 0x2, 0x113, 0x114, 0x7, 0x52, 0x2, 0x2, 0x114, 
    0x3f, 0x3, 0x2, 0x2, 0x2, 0x115, 0x11b, 0x7, 0x52, 0x2, 0x2, 0x116, 
    0x11b, 0x7, 0x51, 0x2, 0x2, 0x117, 0x11b, 0x5, 0x44, 0x23, 0x2, 0x118, 
    0x11b, 0x7, 0x55, 0x2, 0x2, 0x119, 0x11b, 0x3, 0x2, 0x2, 0x2, 0x11a, 
    0x115, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x116, 0x3, 0x2, 0x2, 0x2, 0x11a, 
    0x117, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x118, 0x3, 0x2, 0x2, 0x2, 0x11a, 
    0x119, 0x3, 0x2, 0x2, 0x2, 0x11b, 0x41, 0x3, 0x2, 0x2, 0x2, 0x11c, 0x11d, 
    0x8, 0x22, 0x1, 0x2, 0x11d, 0x11e, 0x7, 0x4e, 0x2, 0x2, 0x11e, 0x150, 
    0x5, 0x42, 0x22, 0x1d, 0x11f, 0x120, 0x7, 0x42, 0x2, 0x2, 0x120, 0x150, 
    0x5, 0x42, 0x22, 0x1c, 0x121, 0x122, 0x7, 0x39, 0x2, 0x2, 0x122, 0x123, 
    0x7, 0x48, 0x2, 0x2, 0x123, 0x124, 0x5, 0x42, 0x22, 0x2, 0x124, 0x125, 
    0x7, 0x49, 0x2, 0x2, 0x125, 0x150, 0x3, 0x2, 0x2, 0x2, 0x126, 0x127, 
    0x7, 0x3a, 0x2, 0x2, 0x127, 0x128, 0x7, 0x48, 0x2, 0x2, 0x128, 0x129, 
    0x5, 0x42, 0x22, 0x2, 0x129, 0x12a, 0x7, 0x49, 0x2, 0x2, 0x12a, 0x150, 
    0x3, 0x2, 0x2, 0x2, 0x12b, 0x12c, 0x7, 0x3b, 0x2, 0x2, 0x12c, 0x12d, 
    0x7, 0x48, 0x2, 0x2, 0x12d, 0x12e, 0x5, 0x42, 0x22, 0x2, 0x12e, 0x12f, 
    0x7, 0x49, 0x2, 0x2, 0x12f, 0x150, 0x3, 0x2, 0x2, 0x2, 0x130, 0x131, 
    0x7, 0x3c, 0x2, 0x2, 0x131, 0x132, 0x7, 0x48, 0x2, 0x2, 0x132, 0x133, 
    0x5, 0x42, 0x22, 0x2, 0x133, 0x134, 0x7, 0x49, 0x2, 0x2, 0x134, 0x150, 
    0x3, 0x2, 0x2, 0x2, 0x135, 0x136, 0x7, 0x3d, 0x2, 0x2, 0x136, 0x137, 
    0x7, 0x48, 0x2, 0x2, 0x137, 0x138, 0x5, 0x42, 0x22, 0x2, 0x138, 0x139, 
    0x7, 0x49, 0x2, 0x2, 0x139, 0x150, 0x3, 0x2, 0x2, 0x2, 0x13a, 0x13b, 
    0x7, 0x3e, 0x2, 0x2, 0x13b, 0x13c, 0x7, 0x48, 0x2, 0x2, 0x13c, 0x13d, 
    0x5, 0x42, 0x22, 0x2, 0x13d, 0x13e, 0x7, 0x49, 0x2, 0x2, 0x13e, 0x150, 
    0x3, 0x2, 0x2, 0x2, 0x13f, 0x140, 0x7, 0x48, 0x2, 0x2, 0x140, 0x141, 
    0x5, 0x42, 0x22, 0x2, 0x141, 0x142, 0x7, 0x49, 0x2, 0x2, 0x142, 0x150, 
    0x3, 0x2, 0x2, 0x2, 0x143, 0x150, 0x5, 0x2c, 0x17, 0x2, 0x144, 0x150, 
    0x5, 0x44, 0x23, 0x2, 0x145, 0x150, 0x7, 0x3, 0x2, 0x2, 0x146, 0x150, 
    0x7, 0x51, 0x2, 0x2, 0x147, 0x150, 0x7, 0x52, 0x2, 0x2, 0x148, 0x150, 
    0x7, 0x55, 0x2, 0x2, 0x149, 0x150, 0x7, 0x54, 0x2, 0x2, 0x14a, 0x14b, 
    0x7, 0x33, 0x2, 0x2, 0x14b, 0x14c, 0x7, 0x48, 0x2, 0x2, 0x14c, 0x14d, 
    0x5, 0x42, 0x22, 0x2, 0x14d, 0x14e, 0x7, 0x49, 0x2, 0x2, 0x14e, 0x150, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x11f, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x121, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x126, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x130, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x135, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x13a, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x13f, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x143, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x144, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x145, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x146, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x147, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x148, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x149, 
    0x3, 0x2, 0x2, 0x2, 0x14f, 0x14a, 0x3, 0x2, 0x2, 0x2, 0x150, 0x174, 
    0x3, 0x2, 0x2, 0x2, 0x151, 0x152, 0xc, 0x15, 0x2, 0x2, 0x152, 0x153, 
    0x9, 0x3, 0x2, 0x2, 0x153, 0x173, 0x5, 0x42, 0x22, 0x16, 0x154, 0x155, 
    0xc, 0x14, 0x2, 0x2, 0x155, 0x156, 0x9, 0x4, 0x2, 0x2, 0x156, 0x173, 
    0x5, 0x42, 0x22, 0x15, 0x157, 0x158, 0xc, 0x13, 0x2, 0x2, 0x158, 0x159, 
    0x9, 0x5, 0x2, 0x2, 0x159, 0x173, 0x5, 0x42, 0x22, 0x14, 0x15a, 0x15b, 
    0xc, 0x12, 0x2, 0x2, 0x15b, 0x15c, 0x9, 0x6, 0x2, 0x2, 0x15c, 0x173, 
    0x5, 0x42, 0x22, 0x13, 0x15d, 0x15e, 0xc, 0x11, 0x2, 0x2, 0x15e, 0x15f, 
    0x9, 0x7, 0x2, 0x2, 0x15f, 0x173, 0x5, 0x42, 0x22, 0x12, 0x160, 0x161, 
    0xc, 0x10, 0x2, 0x2, 0x161, 0x162, 0x7, 0x45, 0x2, 0x2, 0x162, 0x173, 
    0x5, 0x42, 0x22, 0x11, 0x163, 0x164, 0xc, 0xf, 0x2, 0x2, 0x164, 0x165, 
    0x7, 0x3f, 0x2, 0x2, 0x165, 0x173, 0x5, 0x42, 0x22, 0x10, 0x166, 0x167, 
    0xc, 0xe, 0x2, 0x2, 0x167, 0x168, 0x7, 0x2a, 0x2, 0x2, 0x168, 0x169, 
    0x5, 0x42, 0x22, 0x2, 0x169, 0x16a, 0x7, 0x50, 0x2, 0x2, 0x16a, 0x16b, 
    0x5, 0x42, 0x22, 0xf, 0x16b, 0x173, 0x3, 0x2, 0x2, 0x2, 0x16c, 0x16d, 
    0xc, 0xd, 0x2, 0x2, 0x16d, 0x16e, 0x7, 0x50, 0x2, 0x2, 0x16e, 0x173, 
    0x5, 0x42, 0x22, 0xe, 0x16f, 0x170, 0xc, 0xc, 0x2, 0x2, 0x170, 0x171, 
    0x7, 0x4f, 0x2, 0x2, 0x171, 0x173, 0x5, 0x42, 0x22, 0xd, 0x172, 0x151, 
    0x3, 0x2, 0x2, 0x2, 0x172, 0x154, 0x3, 0x2, 0x2, 0x2, 0x172, 0x157, 
    0x3, 0x2, 0x2, 0x2, 0x172, 0x15a, 0x3, 0x2, 0x2, 0x2, 0x172, 0x15d, 
    0x3, 0x2, 0x2, 0x2, 0x172, 0x160, 0x3, 0x2, 0x2, 0x2, 0x172, 0x163, 
    0x3, 0x2, 0x2, 0x2, 0x172, 0x166, 0x3, 0x2, 0x2, 0x2, 0x172, 0x16c, 
    0x3, 0x2, 0x2, 0x2, 0x172, 0x16f, 0x3, 0x2, 0x2, 0x2, 0x173, 0x176, 
    0x3, 0x2, 0x2, 0x2, 0x174, 0x172, 0x3, 0x2, 0x2, 0x2, 0x174, 0x175, 
    0x3, 0x2, 0x2, 0x2, 0x175, 0x43, 0x3, 0x2, 0x2, 0x2, 0x176, 0x174, 0x3, 
    0x2, 0x2, 0x2, 0x177, 0x17e, 0x5, 0x46, 0x24, 0x2, 0x178, 0x17e, 0x5, 
    0x4a, 0x26, 0x2, 0x179, 0x17e, 0x5, 0x48, 0x25, 0x2, 0x17a, 0x17e, 0x5, 
    0x4c, 0x27, 0x2, 0x17b, 0x17e, 0x5, 0x4e, 0x28, 0x2, 0x17c, 0x17e, 0x5, 
    0x50, 0x29, 0x2, 0x17d, 0x177, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x178, 0x3, 
    0x2, 0x2, 0x2, 0x17d, 0x179, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x17a, 0x3, 
    0x2, 0x2, 0x2, 0x17d, 0x17b, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x17c, 0x3, 
    0x2, 0x2, 0x2, 0x17e, 0x45, 0x3, 0x2, 0x2, 0x2, 0x17f, 0x180, 0x7, 0x11, 
    0x2, 0x2, 0x180, 0x181, 0x7, 0x48, 0x2, 0x2, 0x181, 0x182, 0x5, 0x58, 
    0x2d, 0x2, 0x182, 0x183, 0x7, 0x49, 0x2, 0x2, 0x183, 0x47, 0x3, 0x2, 
    0x2, 0x2, 0x184, 0x185, 0x7, 0x13, 0x2, 0x2, 0x185, 0x186, 0x5, 0x56, 
    0x2c, 0x2, 0x186, 0x49, 0x3, 0x2, 0x2, 0x2, 0x187, 0x188, 0x7, 0x15, 
    0x2, 0x2, 0x188, 0x189, 0x5, 0x54, 0x2b, 0x2, 0x189, 0x4b, 0x3, 0x2, 
    0x2, 0x2, 0x18a, 0x18b, 0x7, 0x12, 0x2, 0x2, 0x18b, 0x18c, 0x7, 0x48, 
    0x2, 0x2, 0x18c, 0x191, 0x5, 0x52, 0x2a, 0x2, 0x18d, 0x18e, 0x7, 0xc, 
    0x2, 0x2, 0x18e, 0x190, 0x5, 0x52, 0x2a, 0x2, 0x18f, 0x18d, 0x3, 0x2, 
    0x2, 0x2, 0x190, 0x193, 0x3, 0x2, 0x2, 0x2, 0x191, 0x18f, 0x3, 0x2, 
    0x2, 0x2, 0x191, 0x192, 0x3, 0x2, 0x2, 0x2, 0x192, 0x194, 0x3, 0x2, 
    0x2, 0x2, 0x193, 0x191, 0x3, 0x2, 0x2, 0x2, 0x194, 0x195, 0x7, 0x49, 
    0x2, 0x2, 0x195, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x196, 0x197, 0x7, 0x14, 
    0x2, 0x2, 0x197, 0x198, 0x7, 0x48, 0x2, 0x2, 0x198, 0x19d, 0x5, 0x56, 
    0x2c, 0x2, 0x199, 0x19a, 0x7, 0xc, 0x2, 0x2, 0x19a, 0x19c, 0x5, 0x56, 
    0x2c, 0x2, 0x19b, 0x199, 0x3, 0x2, 0x2, 0x2, 0x19c, 0x19f, 0x3, 0x2, 
    0x2, 0x2, 0x19d, 0x19b, 0x3, 0x2, 0x2, 0x2, 0x19d, 0x19e, 0x3, 0x2, 
    0x2, 0x2, 0x19e, 0x1a0, 0x3, 0x2, 0x2, 0x2, 0x19f, 0x19d, 0x3, 0x2, 
    0x2, 0x2, 0x1a0, 0x1a1, 0x7, 0x49, 0x2, 0x2, 0x1a1, 0x4f, 0x3, 0x2, 
    0x2, 0x2, 0x1a2, 0x1a3, 0x7, 0x16, 0x2, 0x2, 0x1a3, 0x1a4, 0x7, 0x48, 
    0x2, 0x2, 0x1a4, 0x1a9, 0x5, 0x54, 0x2b, 0x2, 0x1a5, 0x1a6, 0x7, 0xc, 
    0x2, 0x2, 0x1a6, 0x1a8, 0x5, 0x54, 0x2b, 0x2, 0x1a7, 0x1a5, 0x3, 0x2, 
    0x2, 0x2, 0x1a8, 0x1ab, 0x3, 0x2, 0x2, 0x2, 0x1a9, 0x1a7, 0x3, 0x2, 
    0x2, 0x2, 0x1a9, 0x1aa, 0x3, 0x2, 0x2, 0x2, 0x1aa, 0x1ac, 0x3, 0x2, 
    0x2, 0x2, 0x1ab, 0x1a9, 0x3, 0x2, 0x2, 0x2, 0x1ac, 0x1ad, 0x7, 0x49, 
    0x2, 0x2, 0x1ad, 0x51, 0x3, 0x2, 0x2, 0x2, 0x1ae, 0x1b4, 0x5, 0x58, 
    0x2d, 0x2, 0x1af, 0x1b0, 0x7, 0x48, 0x2, 0x2, 0x1b0, 0x1b1, 0x5, 0x58, 
    0x2d, 0x2, 0x1b1, 0x1b2, 0x7, 0x49, 0x2, 0x2, 0x1b2, 0x1b4, 0x3, 0x2, 
    0x2, 0x2, 0x1b3, 0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1b3, 0x1af, 0x3, 0x2, 
    0x2, 0x2, 0x1b4, 0x53, 0x3, 0x2, 0x2, 0x2, 0x1b5, 0x1b6, 0x7, 0x48, 
    0x2, 0x2, 0x1b6, 0x1bb, 0x5, 0x56, 0x2c, 0x2, 0x1b7, 0x1b8, 0x7, 0xc, 
    0x2, 0x2, 0x1b8, 0x1ba, 0x5, 0x56, 0x2c, 0x2, 0x1b9, 0x1b7, 0x3, 0x2, 
    0x2, 0x2, 0x1ba, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x1bb, 0x1b9, 0x3, 0x2, 
    0x2, 0x2, 0x1bb, 0x1bc, 0x3, 0x2, 0x2, 0x2, 0x1bc, 0x1be, 0x3, 0x2, 
    0x2, 0x2, 0x1bd, 0x1bb, 0x3, 0x2, 0x2, 0x2, 0x1be, 0x1bf, 0x7, 0x49, 
    0x2, 0x2, 0x1bf, 0x55, 0x3, 0x2, 0x2, 0x2, 0x1c0, 0x1c1, 0x7, 0x48, 
    0x2, 0x2, 0x1c1, 0x1c6, 0x5, 0x58, 0x2d, 0x2, 0x1c2, 0x1c3, 0x7, 0xc, 
    0x2, 0x2, 0x1c3, 0x1c5, 0x5, 0x58, 0x2d, 0x2, 0x1c4, 0x1c2, 0x3, 0x2, 
    0x2, 0x2, 0x1c5, 0x1c8, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c4, 0x3, 0x2, 
    0x2, 0x2, 0x1c6, 0x1c7, 0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1c9, 0x3, 0x2, 
    0x2, 0x2, 0x1c8, 0x1c6, 0x3, 0x2, 0x2, 0x2, 0x1c9, 0x1ca, 0x7, 0x49, 
    0x2, 0x2, 0x1ca, 0x57, 0x3, 0x2, 0x2, 0x2, 0x1cb, 0x1cc, 0x9, 0x8, 0x2, 
    0x2, 0x1cc, 0x1cd, 0x9, 0x8, 0x2, 0x2, 0x1cd, 0x59, 0x3, 0x2, 0x2, 0x2, 
    0x22, 0x5d, 0x67, 0x6c, 0x74, 0x7d, 0x86, 0x8a, 0x8e, 0x92, 0x96, 0x9a, 
    0xb8, 0xc3, 0xcf, 0xd4, 0xdb, 0xe3, 0xeb, 0xf5, 0xfc, 0x102, 0x11a, 
    0x14f, 0x172, 0x174, 0x17d, 0x191, 0x19d, 0x1a9, 0x1b3, 0x1bb, 0x1c6, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

GpuSqlParser::Initializer GpuSqlParser::_init;

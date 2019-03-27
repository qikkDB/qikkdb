
// Generated from /Users/ms/dropdbase_instarea/dropdbase/GpuSqlParser/GpuSqlParser.g4 by ANTLR 4.7.2


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
    setState(93);
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
      setState(90);
      statement();
      setState(95);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(96);
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
    setState(103);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SELECT: {
        enterOuterAlt(_localctx, 1);
        setState(98);
        sqlSelect();
        break;
      }

      case GpuSqlParser::CREATEDB: {
        enterOuterAlt(_localctx, 2);
        setState(99);
        sqlCreateDb();
        break;
      }

      case GpuSqlParser::CREATETABLE: {
        enterOuterAlt(_localctx, 3);
        setState(100);
        sqlCreateTable();
        break;
      }

      case GpuSqlParser::INSERTINTO: {
        enterOuterAlt(_localctx, 4);
        setState(101);
        sqlInsertInto();
        break;
      }

      case GpuSqlParser::SHOWDB:
      case GpuSqlParser::SHOWTB:
      case GpuSqlParser::SHOWCL: {
        enterOuterAlt(_localctx, 5);
        setState(102);
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
    setState(108);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SHOWDB: {
        setState(105);
        showDatabases();
        break;
      }

      case GpuSqlParser::SHOWTB: {
        setState(106);
        showTables();
        break;
      }

      case GpuSqlParser::SHOWCL: {
        setState(107);
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
    setState(110);
    match(GpuSqlParser::SHOWDB);
    setState(111);
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
    setState(113);
    match(GpuSqlParser::SHOWTB);
    setState(116);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(114);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(115);
      database();
    }
    setState(118);
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
    setState(120);
    match(GpuSqlParser::SHOWCL);
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
    table();
    setState(125);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(123);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(124);
      database();
    }
    setState(127);
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
    setState(129);
    match(GpuSqlParser::SELECT);
    setState(130);
    selectColumns();
    setState(131);
    match(GpuSqlParser::FROM);
    setState(132);
    fromTables();
    setState(134);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::JOIN) {
      setState(133);
      joinClauses();
    }
    setState(138);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::WHERE) {
      setState(136);
      match(GpuSqlParser::WHERE);
      setState(137);
      whereClause();
    }
    setState(142);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::GROUPBY) {
      setState(140);
      match(GpuSqlParser::GROUPBY);
      setState(141);
      groupByColumns();
    }
    setState(146);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::ORDERBY) {
      setState(144);
      match(GpuSqlParser::ORDERBY);
      setState(145);
      orderByColumns();
    }
    setState(150);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::LIMIT) {
      setState(148);
      match(GpuSqlParser::LIMIT);
      setState(149);
      limit();
    }
    setState(154);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::OFFSET) {
      setState(152);
      match(GpuSqlParser::OFFSET);
      setState(153);
      offset();
    }
    setState(156);
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
    setState(158);
    match(GpuSqlParser::CREATEDB);
    setState(159);
    database();
    setState(160);
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
    setState(162);
    match(GpuSqlParser::CREATETABLE);
    setState(163);
    table();
    setState(164);
    match(GpuSqlParser::LPAREN);
    setState(165);
    newTableColumns();
    setState(166);
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
    setState(168);
    match(GpuSqlParser::INSERTINTO);
    setState(169);
    table();
    setState(170);
    match(GpuSqlParser::LPAREN);
    setState(171);
    insertIntoColumns();
    setState(172);
    match(GpuSqlParser::RPAREN);
    setState(173);
    match(GpuSqlParser::VALUES);
    setState(174);
    match(GpuSqlParser::LPAREN);
    setState(175);
    insertIntoValues();
    setState(176);
    match(GpuSqlParser::RPAREN);
    setState(177);
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
    setState(179);
    newTableColumn();
    setState(184);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(180);
      match(GpuSqlParser::COMMA);
      setState(181);
      newTableColumn();
      setState(186);
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
    setState(187);
    columnId();
    setState(188);
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
    setState(190);
    selectColumn();
    setState(195);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(191);
      match(GpuSqlParser::COMMA);
      setState(192);
      selectColumn();
      setState(197);
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

tree::TerminalNode* GpuSqlParser::SelectColumnContext::AS() {
  return getToken(GpuSqlParser::AS, 0);
}

GpuSqlParser::AliasContext* GpuSqlParser::SelectColumnContext::alias() {
  return getRuleContext<GpuSqlParser::AliasContext>(0);
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
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(198);
    expression(0);
    setState(201);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(199);
      match(GpuSqlParser::AS);
      setState(200);
      alias();
    }
   
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
    setState(203);
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
    setState(205);
    orderByColumn();
    setState(210);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(206);
      match(GpuSqlParser::COMMA);
      setState(207);
      orderByColumn();
      setState(212);
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
    setState(213);
    columnId();
    setState(215);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::DIR) {
      setState(214);
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
    setState(217);
    columnValue();
    setState(222);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(218);
      match(GpuSqlParser::COMMA);
      setState(219);
      columnValue();
      setState(224);
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
    setState(225);
    columnId();
    setState(230);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(226);
      match(GpuSqlParser::COMMA);
      setState(227);
      columnId();
      setState(232);
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
    setState(233);
    groupByColumn();
    setState(238);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(234);
      match(GpuSqlParser::COMMA);
      setState(235);
      groupByColumn();
      setState(240);
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
    setState(241);
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
    setState(248);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(243);
      column();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(244);
      table();
      setState(245);
      match(GpuSqlParser::DOT);
      setState(246);
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
    setState(250);
    table();
    setState(255);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(251);
      match(GpuSqlParser::COMMA);
      setState(252);
      table();
      setState(257);
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
    setState(259); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(258);
      joinClause();
      setState(261); 
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
    setState(263);
    match(GpuSqlParser::JOIN);
    setState(264);
    joinTable();
    setState(265);
    match(GpuSqlParser::ON);
    setState(266);
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
    setState(268);
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
    setState(270);
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
    setState(272);
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
    setState(274);
    match(GpuSqlParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AliasContext ------------------------------------------------------------------

GpuSqlParser::AliasContext::AliasContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::AliasContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}


size_t GpuSqlParser::AliasContext::getRuleIndex() const {
  return GpuSqlParser::RuleAlias;
}

void GpuSqlParser::AliasContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAlias(this);
}

void GpuSqlParser::AliasContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAlias(this);
}

GpuSqlParser::AliasContext* GpuSqlParser::alias() {
  AliasContext *_localctx = _tracker.createInstance<AliasContext>(_ctx, getState());
  enterRule(_localctx, 58, GpuSqlParser::RuleAlias);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(276);
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
  enterRule(_localctx, 60, GpuSqlParser::RuleLimit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(278);
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
  enterRule(_localctx, 62, GpuSqlParser::RuleOffset);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(280);
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
  enterRule(_localctx, 64, GpuSqlParser::RuleColumnValue);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(287);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::INTLIT: {
        setState(282);
        match(GpuSqlParser::INTLIT);
        break;
      }

      case GpuSqlParser::FLOATLIT: {
        setState(283);
        match(GpuSqlParser::FLOATLIT);
        break;
      }

      case GpuSqlParser::POINT:
      case GpuSqlParser::MULTIPOINT:
      case GpuSqlParser::LINESTRING:
      case GpuSqlParser::MULTILINESTRING:
      case GpuSqlParser::POLYGON:
      case GpuSqlParser::MULTIPOLYGON: {
        setState(284);
        geometry();
        break;
      }

      case GpuSqlParser::STRINGLIT: {
        setState(285);
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
//----------------- BinaryOperationContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::COMMA() {
  return getToken(GpuSqlParser::COMMA, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::POINT() {
  return getToken(GpuSqlParser::POINT, 0);
}

std::vector<GpuSqlParser::ExpressionContext *> GpuSqlParser::BinaryOperationContext::expression() {
  return getRuleContexts<GpuSqlParser::ExpressionContext>();
}

GpuSqlParser::ExpressionContext* GpuSqlParser::BinaryOperationContext::expression(size_t i) {
  return getRuleContext<GpuSqlParser::ExpressionContext>(i);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO_CONTAINS() {
  return getToken(GpuSqlParser::GEO_CONTAINS, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO_INTERSECT() {
  return getToken(GpuSqlParser::GEO_INTERSECT, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO_UNION() {
  return getToken(GpuSqlParser::GEO_UNION, 0);
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
  size_t startState = 66;
  enterRecursionRule(_localctx, 66, GpuSqlParser::RuleExpression, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(368);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(290);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOT);
      setState(291);
      expression(30);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(292);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUS);
      setState(293);
      expression(29);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(294);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::YEAR);
      setState(295);
      match(GpuSqlParser::LPAREN);
      setState(296);
      expression(0);
      setState(297);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(299);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MONTH);
      setState(300);
      match(GpuSqlParser::LPAREN);
      setState(301);
      expression(0);
      setState(302);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(304);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DAY);
      setState(305);
      match(GpuSqlParser::LPAREN);
      setState(306);
      expression(0);
      setState(307);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(309);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::HOUR);
      setState(310);
      match(GpuSqlParser::LPAREN);
      setState(311);
      expression(0);
      setState(312);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(314);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUTE);
      setState(315);
      match(GpuSqlParser::LPAREN);
      setState(316);
      expression(0);
      setState(317);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(319);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SECOND);
      setState(320);
      match(GpuSqlParser::LPAREN);
      setState(321);
      expression(0);
      setState(322);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(324);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POINT);
      setState(325);
      match(GpuSqlParser::LPAREN);
      setState(326);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(327);
      match(GpuSqlParser::COMMA);
      setState(328);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(329);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(331);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_CONTAINS);
      setState(332);
      match(GpuSqlParser::LPAREN);
      setState(333);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(334);
      match(GpuSqlParser::COMMA);
      setState(335);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(336);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 11: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(338);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_INTERSECT);
      setState(339);
      match(GpuSqlParser::LPAREN);
      setState(340);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(341);
      match(GpuSqlParser::COMMA);
      setState(342);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(343);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 12: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(345);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_UNION);
      setState(346);
      match(GpuSqlParser::LPAREN);
      setState(347);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(348);
      match(GpuSqlParser::COMMA);
      setState(349);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(350);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 13: {
      _localctx = _tracker.createInstance<ParenExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(352);
      match(GpuSqlParser::LPAREN);
      setState(353);
      expression(0);
      setState(354);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 14: {
      _localctx = _tracker.createInstance<VarReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(356);
      columnId();
      break;
    }

    case 15: {
      _localctx = _tracker.createInstance<GeoReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(357);
      geometry();
      break;
    }

    case 16: {
      _localctx = _tracker.createInstance<DateTimeLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(358);
      match(GpuSqlParser::DATETIMELIT);
      break;
    }

    case 17: {
      _localctx = _tracker.createInstance<DecimalLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(359);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 18: {
      _localctx = _tracker.createInstance<IntLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(360);
      match(GpuSqlParser::INTLIT);
      break;
    }

    case 19: {
      _localctx = _tracker.createInstance<StringLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(361);
      match(GpuSqlParser::STRINGLIT);
      break;
    }

    case 20: {
      _localctx = _tracker.createInstance<BooleanLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(362);
      match(GpuSqlParser::BOOLEANLIT);
      break;
    }

    case 21: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(363);
      match(GpuSqlParser::AGG);
      setState(364);
      match(GpuSqlParser::LPAREN);
      setState(365);
      expression(0);
      setState(366);
      match(GpuSqlParser::RPAREN);
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(402);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(400);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(370);

          if (!(precpred(_ctx, 22))) throw FailedPredicateException(this, "precpred(_ctx, 22)");
          setState(371);
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
          setState(372);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(23);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(373);

          if (!(precpred(_ctx, 21))) throw FailedPredicateException(this, "precpred(_ctx, 21)");
          setState(374);
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
          setState(375);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(22);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(376);

          if (!(precpred(_ctx, 20))) throw FailedPredicateException(this, "precpred(_ctx, 20)");
          setState(377);
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
          setState(378);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(21);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(379);

          if (!(precpred(_ctx, 19))) throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(380);
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
          setState(381);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(20);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(382);

          if (!(precpred(_ctx, 18))) throw FailedPredicateException(this, "precpred(_ctx, 18)");
          setState(383);
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
          setState(384);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(19);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(385);

          if (!(precpred(_ctx, 17))) throw FailedPredicateException(this, "precpred(_ctx, 17)");
          setState(386);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MODULO);
          setState(387);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(18);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<TernaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(388);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(389);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op = match(GpuSqlParser::BETWEEN);
          setState(390);
          expression(0);
          setState(391);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op2 = match(GpuSqlParser::AND);
          setState(392);
          expression(13);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(394);

          if (!(precpred(_ctx, 11))) throw FailedPredicateException(this, "precpred(_ctx, 11)");
          setState(395);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::AND);
          setState(396);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(12);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(397);

          if (!(precpred(_ctx, 10))) throw FailedPredicateException(this, "precpred(_ctx, 10)");
          setState(398);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::OR);
          setState(399);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(11);
          break;
        }

        } 
      }
      setState(404);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
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
  enterRule(_localctx, 68, GpuSqlParser::RuleGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(411);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::POINT: {
        setState(405);
        pointGeometry();
        break;
      }

      case GpuSqlParser::POLYGON: {
        setState(406);
        polygonGeometry();
        break;
      }

      case GpuSqlParser::LINESTRING: {
        setState(407);
        lineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOINT: {
        setState(408);
        multiPointGeometry();
        break;
      }

      case GpuSqlParser::MULTILINESTRING: {
        setState(409);
        multiLineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOLYGON: {
        setState(410);
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
  enterRule(_localctx, 70, GpuSqlParser::RulePointGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(413);
    match(GpuSqlParser::POINT);
    setState(414);
    match(GpuSqlParser::LPAREN);
    setState(415);
    point();
    setState(416);
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
  enterRule(_localctx, 72, GpuSqlParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(418);
    match(GpuSqlParser::LINESTRING);
    setState(419);
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
  enterRule(_localctx, 74, GpuSqlParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(421);
    match(GpuSqlParser::POLYGON);
    setState(422);
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
  enterRule(_localctx, 76, GpuSqlParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(424);
    match(GpuSqlParser::MULTIPOINT);
    setState(425);
    match(GpuSqlParser::LPAREN);
    setState(426);
    pointOrClosedPoint();
    setState(431);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(427);
      match(GpuSqlParser::COMMA);
      setState(428);
      pointOrClosedPoint();
      setState(433);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(434);
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
  enterRule(_localctx, 78, GpuSqlParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(436);
    match(GpuSqlParser::MULTILINESTRING);
    setState(437);
    match(GpuSqlParser::LPAREN);
    setState(438);
    lineString();
    setState(443);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(439);
      match(GpuSqlParser::COMMA);
      setState(440);
      lineString();
      setState(445);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(446);
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
  enterRule(_localctx, 80, GpuSqlParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(448);
    match(GpuSqlParser::MULTIPOLYGON);
    setState(449);
    match(GpuSqlParser::LPAREN);
    setState(450);
    polygon();
    setState(455);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(451);
      match(GpuSqlParser::COMMA);
      setState(452);
      polygon();
      setState(457);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(458);
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
  enterRule(_localctx, 82, GpuSqlParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(465);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT: {
        enterOuterAlt(_localctx, 1);
        setState(460);
        point();
        break;
      }

      case GpuSqlParser::LPAREN: {
        enterOuterAlt(_localctx, 2);
        setState(461);
        match(GpuSqlParser::LPAREN);
        setState(462);
        point();
        setState(463);
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
  enterRule(_localctx, 84, GpuSqlParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(467);
    match(GpuSqlParser::LPAREN);
    setState(468);
    lineString();
    setState(473);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(469);
      match(GpuSqlParser::COMMA);
      setState(470);
      lineString();
      setState(475);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(476);
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
  enterRule(_localctx, 86, GpuSqlParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(478);
    match(GpuSqlParser::LPAREN);
    setState(479);
    point();
    setState(484);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(480);
      match(GpuSqlParser::COMMA);
      setState(481);
      point();
      setState(486);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(487);
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
  enterRule(_localctx, 88, GpuSqlParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(489);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FLOATLIT

    || _la == GpuSqlParser::INTLIT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(490);
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
    case 33: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool GpuSqlParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 22);
    case 1: return precpred(_ctx, 21);
    case 2: return precpred(_ctx, 20);
    case 3: return precpred(_ctx, 19);
    case 4: return precpred(_ctx, 18);
    case 5: return precpred(_ctx, 17);
    case 6: return precpred(_ctx, 12);
    case 7: return precpred(_ctx, 11);
    case 8: return precpred(_ctx, 10);

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
  "alias", "limit", "offset", "columnValue", "expression", "geometry", "pointGeometry", 
  "lineStringGeometry", "polygonGeometry", "multiPointGeometry", "multiLineStringGeometry", 
  "multiPolygonGeometry", "pointOrClosedPoint", "polygon", "lineString", 
  "point"
};

std::vector<std::string> GpuSqlParser::_literalNames = {
  "", "", "'\n'", "'\r'", "'\r\n'", "", "';'", "'''", "'\"'", "'_'", "':'", 
  "','", "'.'", "", "", "", "'POINT'", "'MULTIPOINT'", "'LINESTRING'", "'MULTILINESTRING'", 
  "'POLYGON'", "'MULTIPOLYGON'", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "'+'", "'-'", 
  "'*'", "'/'", "'%'", "'='", "'!='", "'('", "')'", "'>'", "'<'", "'>='", 
  "'<='", "'!'"
};

std::vector<std::string> GpuSqlParser::_symbolicNames = {
  "", "DATETIMELIT", "LF", "CR", "CRLF", "WS", "SEMICOL", "SQOUTE", "DQOUTE", 
  "UNDERSCORE", "COLON", "COMMA", "DOT", "DATELIT", "TIMELIT", "DATATYPE", 
  "POINT", "MULTIPOINT", "LINESTRING", "MULTILINESTRING", "POLYGON", "MULTIPOLYGON", 
  "INTTYPE", "LONGTYPE", "FLOATTYPE", "DOUBLETYPE", "STRINGTYPE", "BOOLEANTYPE", 
  "POINTTYPE", "POLYTYPE", "INSERTINTO", "CREATEDB", "CREATETABLE", "VALUES", 
  "SELECT", "FROM", "JOIN", "WHERE", "GROUPBY", "AS", "IN", "BETWEEN", "ON", 
  "ORDERBY", "DIR", "LIMIT", "OFFSET", "SHOWDB", "SHOWTB", "SHOWCL", "AGG", 
  "AVG", "SUM", "MIN", "MAX", "COUNT", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", 
  "SECOND", "GEO_CONTAINS", "GEO_INTERSECT", "GEO_UNION", "PLUS", "MINUS", 
  "ASTERISK", "DIVISION", "MODULO", "EQUALS", "NOTEQUALS", "LPAREN", "RPAREN", 
  "GREATER", "LESS", "GREATEREQ", "LESSEQ", "NOT", "OR", "AND", "FLOATLIT", 
  "INTLIT", "ID", "BOOLEANLIT", "STRINGLIT"
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
    0x3, 0x57, 0x1ef, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x2c, 0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x4, 0x2e, 0x9, 0x2e, 0x3, 0x2, 
    0x7, 0x2, 0x5e, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x61, 0xb, 0x2, 0x3, 0x2, 
    0x3, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 
    0x6a, 0xa, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 0x6f, 0xa, 0x4, 
    0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 
    0x77, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x7, 0x3, 0x7, 0x5, 0x7, 0x80, 0xa, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x89, 0xa, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x5, 0x8, 0x8d, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 
    0x91, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x95, 0xa, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x5, 0x8, 0x99, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x9d, 
    0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x7, 0xc, 
    0xb9, 0xa, 0xc, 0xc, 0xc, 0xe, 0xc, 0xbc, 0xb, 0xc, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x7, 0xe, 0xc4, 0xa, 0xe, 0xc, 
    0xe, 0xe, 0xe, 0xc7, 0xb, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x5, 0xf, 
    0xcc, 0xa, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x7, 0x11, 0xd3, 0xa, 0x11, 0xc, 0x11, 0xe, 0x11, 0xd6, 0xb, 0x11, 0x3, 
    0x12, 0x3, 0x12, 0x5, 0x12, 0xda, 0xa, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 
    0x13, 0x7, 0x13, 0xdf, 0xa, 0x13, 0xc, 0x13, 0xe, 0x13, 0xe2, 0xb, 0x13, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x7, 0x14, 0xe7, 0xa, 0x14, 0xc, 0x14, 
    0xe, 0x14, 0xea, 0xb, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x7, 0x15, 
    0xef, 0xa, 0x15, 0xc, 0x15, 0xe, 0x15, 0xf2, 0xb, 0x15, 0x3, 0x16, 0x3, 
    0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 
    0xfb, 0xa, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x7, 0x18, 0x100, 
    0xa, 0x18, 0xc, 0x18, 0xe, 0x18, 0x103, 0xb, 0x18, 0x3, 0x19, 0x6, 0x19, 
    0x106, 0xa, 0x19, 0xd, 0x19, 0xe, 0x19, 0x107, 0x3, 0x1a, 0x3, 0x1a, 
    0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1c, 0x3, 
    0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 
    0x3, 0x20, 0x3, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x3, 0x22, 0x3, 0x22, 0x5, 0x22, 0x122, 0xa, 0x22, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x5, 0x23, 0x173, 0xa, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x7, 0x23, 0x193, 0xa, 0x23, 0xc, 0x23, 0xe, 0x23, 0x196, 0xb, 0x23, 
    0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x5, 
    0x24, 0x19e, 0xa, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 
    0x3, 0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x27, 0x3, 0x27, 0x3, 
    0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x7, 0x28, 
    0x1b0, 0xa, 0x28, 0xc, 0x28, 0xe, 0x28, 0x1b3, 0xb, 0x28, 0x3, 0x28, 
    0x3, 0x28, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x7, 
    0x29, 0x1bc, 0xa, 0x29, 0xc, 0x29, 0xe, 0x29, 0x1bf, 0xb, 0x29, 0x3, 
    0x29, 0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 
    0x7, 0x2a, 0x1c8, 0xa, 0x2a, 0xc, 0x2a, 0xe, 0x2a, 0x1cb, 0xb, 0x2a, 
    0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 
    0x2b, 0x5, 0x2b, 0x1d4, 0xa, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 
    0x3, 0x2c, 0x7, 0x2c, 0x1da, 0xa, 0x2c, 0xc, 0x2c, 0xe, 0x2c, 0x1dd, 
    0xb, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 
    0x2d, 0x7, 0x2d, 0x1e5, 0xa, 0x2d, 0xc, 0x2d, 0xe, 0x2d, 0x1e8, 0xb, 
    0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 
    0x2, 0x3, 0x44, 0x2f, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 
    0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 
    0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 
    0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 
    0x2, 0x9, 0x4, 0x2, 0x25, 0x25, 0x2a, 0x2a, 0x3, 0x2, 0x45, 0x46, 0x3, 
    0x2, 0x43, 0x44, 0x3, 0x2, 0x4c, 0x4d, 0x3, 0x2, 0x4e, 0x4f, 0x3, 0x2, 
    0x48, 0x49, 0x3, 0x2, 0x53, 0x54, 0x2, 0x207, 0x2, 0x5f, 0x3, 0x2, 0x2, 
    0x2, 0x4, 0x69, 0x3, 0x2, 0x2, 0x2, 0x6, 0x6e, 0x3, 0x2, 0x2, 0x2, 0x8, 
    0x70, 0x3, 0x2, 0x2, 0x2, 0xa, 0x73, 0x3, 0x2, 0x2, 0x2, 0xc, 0x7a, 
    0x3, 0x2, 0x2, 0x2, 0xe, 0x83, 0x3, 0x2, 0x2, 0x2, 0x10, 0xa0, 0x3, 
    0x2, 0x2, 0x2, 0x12, 0xa4, 0x3, 0x2, 0x2, 0x2, 0x14, 0xaa, 0x3, 0x2, 
    0x2, 0x2, 0x16, 0xb5, 0x3, 0x2, 0x2, 0x2, 0x18, 0xbd, 0x3, 0x2, 0x2, 
    0x2, 0x1a, 0xc0, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xc8, 0x3, 0x2, 0x2, 0x2, 
    0x1e, 0xcd, 0x3, 0x2, 0x2, 0x2, 0x20, 0xcf, 0x3, 0x2, 0x2, 0x2, 0x22, 
    0xd7, 0x3, 0x2, 0x2, 0x2, 0x24, 0xdb, 0x3, 0x2, 0x2, 0x2, 0x26, 0xe3, 
    0x3, 0x2, 0x2, 0x2, 0x28, 0xeb, 0x3, 0x2, 0x2, 0x2, 0x2a, 0xf3, 0x3, 
    0x2, 0x2, 0x2, 0x2c, 0xfa, 0x3, 0x2, 0x2, 0x2, 0x2e, 0xfc, 0x3, 0x2, 
    0x2, 0x2, 0x30, 0x105, 0x3, 0x2, 0x2, 0x2, 0x32, 0x109, 0x3, 0x2, 0x2, 
    0x2, 0x34, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x36, 0x110, 0x3, 0x2, 0x2, 0x2, 
    0x38, 0x112, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x114, 0x3, 0x2, 0x2, 0x2, 0x3c, 
    0x116, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x118, 0x3, 0x2, 0x2, 0x2, 0x40, 0x11a, 
    0x3, 0x2, 0x2, 0x2, 0x42, 0x121, 0x3, 0x2, 0x2, 0x2, 0x44, 0x172, 0x3, 
    0x2, 0x2, 0x2, 0x46, 0x19d, 0x3, 0x2, 0x2, 0x2, 0x48, 0x19f, 0x3, 0x2, 
    0x2, 0x2, 0x4a, 0x1a4, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x1a7, 0x3, 0x2, 0x2, 
    0x2, 0x4e, 0x1aa, 0x3, 0x2, 0x2, 0x2, 0x50, 0x1b6, 0x3, 0x2, 0x2, 0x2, 
    0x52, 0x1c2, 0x3, 0x2, 0x2, 0x2, 0x54, 0x1d3, 0x3, 0x2, 0x2, 0x2, 0x56, 
    0x1d5, 0x3, 0x2, 0x2, 0x2, 0x58, 0x1e0, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x1eb, 
    0x3, 0x2, 0x2, 0x2, 0x5c, 0x5e, 0x5, 0x4, 0x3, 0x2, 0x5d, 0x5c, 0x3, 
    0x2, 0x2, 0x2, 0x5e, 0x61, 0x3, 0x2, 0x2, 0x2, 0x5f, 0x5d, 0x3, 0x2, 
    0x2, 0x2, 0x5f, 0x60, 0x3, 0x2, 0x2, 0x2, 0x60, 0x62, 0x3, 0x2, 0x2, 
    0x2, 0x61, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x62, 0x63, 0x7, 0x2, 0x2, 0x3, 
    0x63, 0x3, 0x3, 0x2, 0x2, 0x2, 0x64, 0x6a, 0x5, 0xe, 0x8, 0x2, 0x65, 
    0x6a, 0x5, 0x10, 0x9, 0x2, 0x66, 0x6a, 0x5, 0x12, 0xa, 0x2, 0x67, 0x6a, 
    0x5, 0x14, 0xb, 0x2, 0x68, 0x6a, 0x5, 0x6, 0x4, 0x2, 0x69, 0x64, 0x3, 
    0x2, 0x2, 0x2, 0x69, 0x65, 0x3, 0x2, 0x2, 0x2, 0x69, 0x66, 0x3, 0x2, 
    0x2, 0x2, 0x69, 0x67, 0x3, 0x2, 0x2, 0x2, 0x69, 0x68, 0x3, 0x2, 0x2, 
    0x2, 0x6a, 0x5, 0x3, 0x2, 0x2, 0x2, 0x6b, 0x6f, 0x5, 0x8, 0x5, 0x2, 
    0x6c, 0x6f, 0x5, 0xa, 0x6, 0x2, 0x6d, 0x6f, 0x5, 0xc, 0x7, 0x2, 0x6e, 
    0x6b, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x6d, 
    0x3, 0x2, 0x2, 0x2, 0x6f, 0x7, 0x3, 0x2, 0x2, 0x2, 0x70, 0x71, 0x7, 
    0x31, 0x2, 0x2, 0x71, 0x72, 0x7, 0x8, 0x2, 0x2, 0x72, 0x9, 0x3, 0x2, 
    0x2, 0x2, 0x73, 0x76, 0x7, 0x32, 0x2, 0x2, 0x74, 0x75, 0x9, 0x2, 0x2, 
    0x2, 0x75, 0x77, 0x5, 0x3a, 0x1e, 0x2, 0x76, 0x74, 0x3, 0x2, 0x2, 0x2, 
    0x76, 0x77, 0x3, 0x2, 0x2, 0x2, 0x77, 0x78, 0x3, 0x2, 0x2, 0x2, 0x78, 
    0x79, 0x7, 0x8, 0x2, 0x2, 0x79, 0xb, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x7b, 
    0x7, 0x33, 0x2, 0x2, 0x7b, 0x7c, 0x9, 0x2, 0x2, 0x2, 0x7c, 0x7f, 0x5, 
    0x36, 0x1c, 0x2, 0x7d, 0x7e, 0x9, 0x2, 0x2, 0x2, 0x7e, 0x80, 0x5, 0x3a, 
    0x1e, 0x2, 0x7f, 0x7d, 0x3, 0x2, 0x2, 0x2, 0x7f, 0x80, 0x3, 0x2, 0x2, 
    0x2, 0x80, 0x81, 0x3, 0x2, 0x2, 0x2, 0x81, 0x82, 0x7, 0x8, 0x2, 0x2, 
    0x82, 0xd, 0x3, 0x2, 0x2, 0x2, 0x83, 0x84, 0x7, 0x24, 0x2, 0x2, 0x84, 
    0x85, 0x5, 0x1a, 0xe, 0x2, 0x85, 0x86, 0x7, 0x25, 0x2, 0x2, 0x86, 0x88, 
    0x5, 0x2e, 0x18, 0x2, 0x87, 0x89, 0x5, 0x30, 0x19, 0x2, 0x88, 0x87, 
    0x3, 0x2, 0x2, 0x2, 0x88, 0x89, 0x3, 0x2, 0x2, 0x2, 0x89, 0x8c, 0x3, 
    0x2, 0x2, 0x2, 0x8a, 0x8b, 0x7, 0x27, 0x2, 0x2, 0x8b, 0x8d, 0x5, 0x1e, 
    0x10, 0x2, 0x8c, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x8d, 0x3, 0x2, 0x2, 
    0x2, 0x8d, 0x90, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x8f, 0x7, 0x28, 0x2, 0x2, 
    0x8f, 0x91, 0x5, 0x28, 0x15, 0x2, 0x90, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x90, 
    0x91, 0x3, 0x2, 0x2, 0x2, 0x91, 0x94, 0x3, 0x2, 0x2, 0x2, 0x92, 0x93, 
    0x7, 0x2d, 0x2, 0x2, 0x93, 0x95, 0x5, 0x20, 0x11, 0x2, 0x94, 0x92, 0x3, 
    0x2, 0x2, 0x2, 0x94, 0x95, 0x3, 0x2, 0x2, 0x2, 0x95, 0x98, 0x3, 0x2, 
    0x2, 0x2, 0x96, 0x97, 0x7, 0x2f, 0x2, 0x2, 0x97, 0x99, 0x5, 0x3e, 0x20, 
    0x2, 0x98, 0x96, 0x3, 0x2, 0x2, 0x2, 0x98, 0x99, 0x3, 0x2, 0x2, 0x2, 
    0x99, 0x9c, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x9b, 0x7, 0x30, 0x2, 0x2, 0x9b, 
    0x9d, 0x5, 0x40, 0x21, 0x2, 0x9c, 0x9a, 0x3, 0x2, 0x2, 0x2, 0x9c, 0x9d, 
    0x3, 0x2, 0x2, 0x2, 0x9d, 0x9e, 0x3, 0x2, 0x2, 0x2, 0x9e, 0x9f, 0x7, 
    0x8, 0x2, 0x2, 0x9f, 0xf, 0x3, 0x2, 0x2, 0x2, 0xa0, 0xa1, 0x7, 0x21, 
    0x2, 0x2, 0xa1, 0xa2, 0x5, 0x3a, 0x1e, 0x2, 0xa2, 0xa3, 0x7, 0x8, 0x2, 
    0x2, 0xa3, 0x11, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa5, 0x7, 0x22, 0x2, 0x2, 
    0xa5, 0xa6, 0x5, 0x36, 0x1c, 0x2, 0xa6, 0xa7, 0x7, 0x4a, 0x2, 0x2, 0xa7, 
    0xa8, 0x5, 0x16, 0xc, 0x2, 0xa8, 0xa9, 0x7, 0x4b, 0x2, 0x2, 0xa9, 0x13, 
    0x3, 0x2, 0x2, 0x2, 0xaa, 0xab, 0x7, 0x20, 0x2, 0x2, 0xab, 0xac, 0x5, 
    0x36, 0x1c, 0x2, 0xac, 0xad, 0x7, 0x4a, 0x2, 0x2, 0xad, 0xae, 0x5, 0x26, 
    0x14, 0x2, 0xae, 0xaf, 0x7, 0x4b, 0x2, 0x2, 0xaf, 0xb0, 0x7, 0x23, 0x2, 
    0x2, 0xb0, 0xb1, 0x7, 0x4a, 0x2, 0x2, 0xb1, 0xb2, 0x5, 0x24, 0x13, 0x2, 
    0xb2, 0xb3, 0x7, 0x4b, 0x2, 0x2, 0xb3, 0xb4, 0x7, 0x8, 0x2, 0x2, 0xb4, 
    0x15, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xba, 0x5, 0x18, 0xd, 0x2, 0xb6, 0xb7, 
    0x7, 0xd, 0x2, 0x2, 0xb7, 0xb9, 0x5, 0x18, 0xd, 0x2, 0xb8, 0xb6, 0x3, 
    0x2, 0x2, 0x2, 0xb9, 0xbc, 0x3, 0x2, 0x2, 0x2, 0xba, 0xb8, 0x3, 0x2, 
    0x2, 0x2, 0xba, 0xbb, 0x3, 0x2, 0x2, 0x2, 0xbb, 0x17, 0x3, 0x2, 0x2, 
    0x2, 0xbc, 0xba, 0x3, 0x2, 0x2, 0x2, 0xbd, 0xbe, 0x5, 0x2c, 0x17, 0x2, 
    0xbe, 0xbf, 0x7, 0x11, 0x2, 0x2, 0xbf, 0x19, 0x3, 0x2, 0x2, 0x2, 0xc0, 
    0xc5, 0x5, 0x1c, 0xf, 0x2, 0xc1, 0xc2, 0x7, 0xd, 0x2, 0x2, 0xc2, 0xc4, 
    0x5, 0x1c, 0xf, 0x2, 0xc3, 0xc1, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc7, 0x3, 
    0x2, 0x2, 0x2, 0xc5, 0xc3, 0x3, 0x2, 0x2, 0x2, 0xc5, 0xc6, 0x3, 0x2, 
    0x2, 0x2, 0xc6, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xc7, 0xc5, 0x3, 0x2, 0x2, 
    0x2, 0xc8, 0xcb, 0x5, 0x44, 0x23, 0x2, 0xc9, 0xca, 0x7, 0x29, 0x2, 0x2, 
    0xca, 0xcc, 0x5, 0x3c, 0x1f, 0x2, 0xcb, 0xc9, 0x3, 0x2, 0x2, 0x2, 0xcb, 
    0xcc, 0x3, 0x2, 0x2, 0x2, 0xcc, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xcd, 0xce, 
    0x5, 0x44, 0x23, 0x2, 0xce, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xcf, 0xd4, 0x5, 
    0x22, 0x12, 0x2, 0xd0, 0xd1, 0x7, 0xd, 0x2, 0x2, 0xd1, 0xd3, 0x5, 0x22, 
    0x12, 0x2, 0xd2, 0xd0, 0x3, 0x2, 0x2, 0x2, 0xd3, 0xd6, 0x3, 0x2, 0x2, 
    0x2, 0xd4, 0xd2, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd5, 0x3, 0x2, 0x2, 0x2, 
    0xd5, 0x21, 0x3, 0x2, 0x2, 0x2, 0xd6, 0xd4, 0x3, 0x2, 0x2, 0x2, 0xd7, 
    0xd9, 0x5, 0x2c, 0x17, 0x2, 0xd8, 0xda, 0x7, 0x2e, 0x2, 0x2, 0xd9, 0xd8, 
    0x3, 0x2, 0x2, 0x2, 0xd9, 0xda, 0x3, 0x2, 0x2, 0x2, 0xda, 0x23, 0x3, 
    0x2, 0x2, 0x2, 0xdb, 0xe0, 0x5, 0x42, 0x22, 0x2, 0xdc, 0xdd, 0x7, 0xd, 
    0x2, 0x2, 0xdd, 0xdf, 0x5, 0x42, 0x22, 0x2, 0xde, 0xdc, 0x3, 0x2, 0x2, 
    0x2, 0xdf, 0xe2, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xde, 0x3, 0x2, 0x2, 0x2, 
    0xe0, 0xe1, 0x3, 0x2, 0x2, 0x2, 0xe1, 0x25, 0x3, 0x2, 0x2, 0x2, 0xe2, 
    0xe0, 0x3, 0x2, 0x2, 0x2, 0xe3, 0xe8, 0x5, 0x2c, 0x17, 0x2, 0xe4, 0xe5, 
    0x7, 0xd, 0x2, 0x2, 0xe5, 0xe7, 0x5, 0x2c, 0x17, 0x2, 0xe6, 0xe4, 0x3, 
    0x2, 0x2, 0x2, 0xe7, 0xea, 0x3, 0x2, 0x2, 0x2, 0xe8, 0xe6, 0x3, 0x2, 
    0x2, 0x2, 0xe8, 0xe9, 0x3, 0x2, 0x2, 0x2, 0xe9, 0x27, 0x3, 0x2, 0x2, 
    0x2, 0xea, 0xe8, 0x3, 0x2, 0x2, 0x2, 0xeb, 0xf0, 0x5, 0x2a, 0x16, 0x2, 
    0xec, 0xed, 0x7, 0xd, 0x2, 0x2, 0xed, 0xef, 0x5, 0x2a, 0x16, 0x2, 0xee, 
    0xec, 0x3, 0x2, 0x2, 0x2, 0xef, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xee, 
    0x3, 0x2, 0x2, 0x2, 0xf0, 0xf1, 0x3, 0x2, 0x2, 0x2, 0xf1, 0x29, 0x3, 
    0x2, 0x2, 0x2, 0xf2, 0xf0, 0x3, 0x2, 0x2, 0x2, 0xf3, 0xf4, 0x5, 0x44, 
    0x23, 0x2, 0xf4, 0x2b, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xfb, 0x5, 0x38, 0x1d, 
    0x2, 0xf6, 0xf7, 0x5, 0x36, 0x1c, 0x2, 0xf7, 0xf8, 0x7, 0xe, 0x2, 0x2, 
    0xf8, 0xf9, 0x5, 0x38, 0x1d, 0x2, 0xf9, 0xfb, 0x3, 0x2, 0x2, 0x2, 0xfa, 
    0xf5, 0x3, 0x2, 0x2, 0x2, 0xfa, 0xf6, 0x3, 0x2, 0x2, 0x2, 0xfb, 0x2d, 
    0x3, 0x2, 0x2, 0x2, 0xfc, 0x101, 0x5, 0x36, 0x1c, 0x2, 0xfd, 0xfe, 0x7, 
    0xd, 0x2, 0x2, 0xfe, 0x100, 0x5, 0x36, 0x1c, 0x2, 0xff, 0xfd, 0x3, 0x2, 
    0x2, 0x2, 0x100, 0x103, 0x3, 0x2, 0x2, 0x2, 0x101, 0xff, 0x3, 0x2, 0x2, 
    0x2, 0x101, 0x102, 0x3, 0x2, 0x2, 0x2, 0x102, 0x2f, 0x3, 0x2, 0x2, 0x2, 
    0x103, 0x101, 0x3, 0x2, 0x2, 0x2, 0x104, 0x106, 0x5, 0x32, 0x1a, 0x2, 
    0x105, 0x104, 0x3, 0x2, 0x2, 0x2, 0x106, 0x107, 0x3, 0x2, 0x2, 0x2, 
    0x107, 0x105, 0x3, 0x2, 0x2, 0x2, 0x107, 0x108, 0x3, 0x2, 0x2, 0x2, 
    0x108, 0x31, 0x3, 0x2, 0x2, 0x2, 0x109, 0x10a, 0x7, 0x26, 0x2, 0x2, 
    0x10a, 0x10b, 0x5, 0x34, 0x1b, 0x2, 0x10b, 0x10c, 0x7, 0x2c, 0x2, 0x2, 
    0x10c, 0x10d, 0x5, 0x44, 0x23, 0x2, 0x10d, 0x33, 0x3, 0x2, 0x2, 0x2, 
    0x10e, 0x10f, 0x5, 0x36, 0x1c, 0x2, 0x10f, 0x35, 0x3, 0x2, 0x2, 0x2, 
    0x110, 0x111, 0x7, 0x55, 0x2, 0x2, 0x111, 0x37, 0x3, 0x2, 0x2, 0x2, 
    0x112, 0x113, 0x7, 0x55, 0x2, 0x2, 0x113, 0x39, 0x3, 0x2, 0x2, 0x2, 
    0x114, 0x115, 0x7, 0x55, 0x2, 0x2, 0x115, 0x3b, 0x3, 0x2, 0x2, 0x2, 
    0x116, 0x117, 0x7, 0x55, 0x2, 0x2, 0x117, 0x3d, 0x3, 0x2, 0x2, 0x2, 
    0x118, 0x119, 0x7, 0x54, 0x2, 0x2, 0x119, 0x3f, 0x3, 0x2, 0x2, 0x2, 
    0x11a, 0x11b, 0x7, 0x54, 0x2, 0x2, 0x11b, 0x41, 0x3, 0x2, 0x2, 0x2, 
    0x11c, 0x122, 0x7, 0x54, 0x2, 0x2, 0x11d, 0x122, 0x7, 0x53, 0x2, 0x2, 
    0x11e, 0x122, 0x5, 0x46, 0x24, 0x2, 0x11f, 0x122, 0x7, 0x57, 0x2, 0x2, 
    0x120, 0x122, 0x3, 0x2, 0x2, 0x2, 0x121, 0x11c, 0x3, 0x2, 0x2, 0x2, 
    0x121, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x121, 0x11e, 0x3, 0x2, 0x2, 0x2, 
    0x121, 0x11f, 0x3, 0x2, 0x2, 0x2, 0x121, 0x120, 0x3, 0x2, 0x2, 0x2, 
    0x122, 0x43, 0x3, 0x2, 0x2, 0x2, 0x123, 0x124, 0x8, 0x23, 0x1, 0x2, 
    0x124, 0x125, 0x7, 0x50, 0x2, 0x2, 0x125, 0x173, 0x5, 0x44, 0x23, 0x20, 
    0x126, 0x127, 0x7, 0x44, 0x2, 0x2, 0x127, 0x173, 0x5, 0x44, 0x23, 0x1f, 
    0x128, 0x129, 0x7, 0x3a, 0x2, 0x2, 0x129, 0x12a, 0x7, 0x4a, 0x2, 0x2, 
    0x12a, 0x12b, 0x5, 0x44, 0x23, 0x2, 0x12b, 0x12c, 0x7, 0x4b, 0x2, 0x2, 
    0x12c, 0x173, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x12e, 0x7, 0x3b, 0x2, 0x2, 
    0x12e, 0x12f, 0x7, 0x4a, 0x2, 0x2, 0x12f, 0x130, 0x5, 0x44, 0x23, 0x2, 
    0x130, 0x131, 0x7, 0x4b, 0x2, 0x2, 0x131, 0x173, 0x3, 0x2, 0x2, 0x2, 
    0x132, 0x133, 0x7, 0x3c, 0x2, 0x2, 0x133, 0x134, 0x7, 0x4a, 0x2, 0x2, 
    0x134, 0x135, 0x5, 0x44, 0x23, 0x2, 0x135, 0x136, 0x7, 0x4b, 0x2, 0x2, 
    0x136, 0x173, 0x3, 0x2, 0x2, 0x2, 0x137, 0x138, 0x7, 0x3d, 0x2, 0x2, 
    0x138, 0x139, 0x7, 0x4a, 0x2, 0x2, 0x139, 0x13a, 0x5, 0x44, 0x23, 0x2, 
    0x13a, 0x13b, 0x7, 0x4b, 0x2, 0x2, 0x13b, 0x173, 0x3, 0x2, 0x2, 0x2, 
    0x13c, 0x13d, 0x7, 0x3e, 0x2, 0x2, 0x13d, 0x13e, 0x7, 0x4a, 0x2, 0x2, 
    0x13e, 0x13f, 0x5, 0x44, 0x23, 0x2, 0x13f, 0x140, 0x7, 0x4b, 0x2, 0x2, 
    0x140, 0x173, 0x3, 0x2, 0x2, 0x2, 0x141, 0x142, 0x7, 0x3f, 0x2, 0x2, 
    0x142, 0x143, 0x7, 0x4a, 0x2, 0x2, 0x143, 0x144, 0x5, 0x44, 0x23, 0x2, 
    0x144, 0x145, 0x7, 0x4b, 0x2, 0x2, 0x145, 0x173, 0x3, 0x2, 0x2, 0x2, 
    0x146, 0x147, 0x7, 0x12, 0x2, 0x2, 0x147, 0x148, 0x7, 0x4a, 0x2, 0x2, 
    0x148, 0x149, 0x5, 0x44, 0x23, 0x2, 0x149, 0x14a, 0x7, 0xd, 0x2, 0x2, 
    0x14a, 0x14b, 0x5, 0x44, 0x23, 0x2, 0x14b, 0x14c, 0x7, 0x4b, 0x2, 0x2, 
    0x14c, 0x173, 0x3, 0x2, 0x2, 0x2, 0x14d, 0x14e, 0x7, 0x40, 0x2, 0x2, 
    0x14e, 0x14f, 0x7, 0x4a, 0x2, 0x2, 0x14f, 0x150, 0x5, 0x44, 0x23, 0x2, 
    0x150, 0x151, 0x7, 0xd, 0x2, 0x2, 0x151, 0x152, 0x5, 0x44, 0x23, 0x2, 
    0x152, 0x153, 0x7, 0x4b, 0x2, 0x2, 0x153, 0x173, 0x3, 0x2, 0x2, 0x2, 
    0x154, 0x155, 0x7, 0x41, 0x2, 0x2, 0x155, 0x156, 0x7, 0x4a, 0x2, 0x2, 
    0x156, 0x157, 0x5, 0x44, 0x23, 0x2, 0x157, 0x158, 0x7, 0xd, 0x2, 0x2, 
    0x158, 0x159, 0x5, 0x44, 0x23, 0x2, 0x159, 0x15a, 0x7, 0x4b, 0x2, 0x2, 
    0x15a, 0x173, 0x3, 0x2, 0x2, 0x2, 0x15b, 0x15c, 0x7, 0x42, 0x2, 0x2, 
    0x15c, 0x15d, 0x7, 0x4a, 0x2, 0x2, 0x15d, 0x15e, 0x5, 0x44, 0x23, 0x2, 
    0x15e, 0x15f, 0x7, 0xd, 0x2, 0x2, 0x15f, 0x160, 0x5, 0x44, 0x23, 0x2, 
    0x160, 0x161, 0x7, 0x4b, 0x2, 0x2, 0x161, 0x173, 0x3, 0x2, 0x2, 0x2, 
    0x162, 0x163, 0x7, 0x4a, 0x2, 0x2, 0x163, 0x164, 0x5, 0x44, 0x23, 0x2, 
    0x164, 0x165, 0x7, 0x4b, 0x2, 0x2, 0x165, 0x173, 0x3, 0x2, 0x2, 0x2, 
    0x166, 0x173, 0x5, 0x2c, 0x17, 0x2, 0x167, 0x173, 0x5, 0x46, 0x24, 0x2, 
    0x168, 0x173, 0x7, 0x3, 0x2, 0x2, 0x169, 0x173, 0x7, 0x53, 0x2, 0x2, 
    0x16a, 0x173, 0x7, 0x54, 0x2, 0x2, 0x16b, 0x173, 0x7, 0x57, 0x2, 0x2, 
    0x16c, 0x173, 0x7, 0x56, 0x2, 0x2, 0x16d, 0x16e, 0x7, 0x34, 0x2, 0x2, 
    0x16e, 0x16f, 0x7, 0x4a, 0x2, 0x2, 0x16f, 0x170, 0x5, 0x44, 0x23, 0x2, 
    0x170, 0x171, 0x7, 0x4b, 0x2, 0x2, 0x171, 0x173, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x123, 0x3, 0x2, 0x2, 0x2, 0x172, 0x126, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x128, 0x3, 0x2, 0x2, 0x2, 0x172, 0x12d, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x132, 0x3, 0x2, 0x2, 0x2, 0x172, 0x137, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x13c, 0x3, 0x2, 0x2, 0x2, 0x172, 0x141, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x146, 0x3, 0x2, 0x2, 0x2, 0x172, 0x14d, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x154, 0x3, 0x2, 0x2, 0x2, 0x172, 0x15b, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x162, 0x3, 0x2, 0x2, 0x2, 0x172, 0x166, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x167, 0x3, 0x2, 0x2, 0x2, 0x172, 0x168, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x169, 0x3, 0x2, 0x2, 0x2, 0x172, 0x16a, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x172, 0x16c, 0x3, 0x2, 0x2, 0x2, 
    0x172, 0x16d, 0x3, 0x2, 0x2, 0x2, 0x173, 0x194, 0x3, 0x2, 0x2, 0x2, 
    0x174, 0x175, 0xc, 0x18, 0x2, 0x2, 0x175, 0x176, 0x9, 0x3, 0x2, 0x2, 
    0x176, 0x193, 0x5, 0x44, 0x23, 0x19, 0x177, 0x178, 0xc, 0x17, 0x2, 0x2, 
    0x178, 0x179, 0x9, 0x4, 0x2, 0x2, 0x179, 0x193, 0x5, 0x44, 0x23, 0x18, 
    0x17a, 0x17b, 0xc, 0x16, 0x2, 0x2, 0x17b, 0x17c, 0x9, 0x5, 0x2, 0x2, 
    0x17c, 0x193, 0x5, 0x44, 0x23, 0x17, 0x17d, 0x17e, 0xc, 0x15, 0x2, 0x2, 
    0x17e, 0x17f, 0x9, 0x6, 0x2, 0x2, 0x17f, 0x193, 0x5, 0x44, 0x23, 0x16, 
    0x180, 0x181, 0xc, 0x14, 0x2, 0x2, 0x181, 0x182, 0x9, 0x7, 0x2, 0x2, 
    0x182, 0x193, 0x5, 0x44, 0x23, 0x15, 0x183, 0x184, 0xc, 0x13, 0x2, 0x2, 
    0x184, 0x185, 0x7, 0x47, 0x2, 0x2, 0x185, 0x193, 0x5, 0x44, 0x23, 0x14, 
    0x186, 0x187, 0xc, 0xe, 0x2, 0x2, 0x187, 0x188, 0x7, 0x2b, 0x2, 0x2, 
    0x188, 0x189, 0x5, 0x44, 0x23, 0x2, 0x189, 0x18a, 0x7, 0x52, 0x2, 0x2, 
    0x18a, 0x18b, 0x5, 0x44, 0x23, 0xf, 0x18b, 0x193, 0x3, 0x2, 0x2, 0x2, 
    0x18c, 0x18d, 0xc, 0xd, 0x2, 0x2, 0x18d, 0x18e, 0x7, 0x52, 0x2, 0x2, 
    0x18e, 0x193, 0x5, 0x44, 0x23, 0xe, 0x18f, 0x190, 0xc, 0xc, 0x2, 0x2, 
    0x190, 0x191, 0x7, 0x51, 0x2, 0x2, 0x191, 0x193, 0x5, 0x44, 0x23, 0xd, 
    0x192, 0x174, 0x3, 0x2, 0x2, 0x2, 0x192, 0x177, 0x3, 0x2, 0x2, 0x2, 
    0x192, 0x17a, 0x3, 0x2, 0x2, 0x2, 0x192, 0x17d, 0x3, 0x2, 0x2, 0x2, 
    0x192, 0x180, 0x3, 0x2, 0x2, 0x2, 0x192, 0x183, 0x3, 0x2, 0x2, 0x2, 
    0x192, 0x186, 0x3, 0x2, 0x2, 0x2, 0x192, 0x18c, 0x3, 0x2, 0x2, 0x2, 
    0x192, 0x18f, 0x3, 0x2, 0x2, 0x2, 0x193, 0x196, 0x3, 0x2, 0x2, 0x2, 
    0x194, 0x192, 0x3, 0x2, 0x2, 0x2, 0x194, 0x195, 0x3, 0x2, 0x2, 0x2, 
    0x195, 0x45, 0x3, 0x2, 0x2, 0x2, 0x196, 0x194, 0x3, 0x2, 0x2, 0x2, 0x197, 
    0x19e, 0x5, 0x48, 0x25, 0x2, 0x198, 0x19e, 0x5, 0x4c, 0x27, 0x2, 0x199, 
    0x19e, 0x5, 0x4a, 0x26, 0x2, 0x19a, 0x19e, 0x5, 0x4e, 0x28, 0x2, 0x19b, 
    0x19e, 0x5, 0x50, 0x29, 0x2, 0x19c, 0x19e, 0x5, 0x52, 0x2a, 0x2, 0x19d, 
    0x197, 0x3, 0x2, 0x2, 0x2, 0x19d, 0x198, 0x3, 0x2, 0x2, 0x2, 0x19d, 
    0x199, 0x3, 0x2, 0x2, 0x2, 0x19d, 0x19a, 0x3, 0x2, 0x2, 0x2, 0x19d, 
    0x19b, 0x3, 0x2, 0x2, 0x2, 0x19d, 0x19c, 0x3, 0x2, 0x2, 0x2, 0x19e, 
    0x47, 0x3, 0x2, 0x2, 0x2, 0x19f, 0x1a0, 0x7, 0x12, 0x2, 0x2, 0x1a0, 
    0x1a1, 0x7, 0x4a, 0x2, 0x2, 0x1a1, 0x1a2, 0x5, 0x5a, 0x2e, 0x2, 0x1a2, 
    0x1a3, 0x7, 0x4b, 0x2, 0x2, 0x1a3, 0x49, 0x3, 0x2, 0x2, 0x2, 0x1a4, 
    0x1a5, 0x7, 0x14, 0x2, 0x2, 0x1a5, 0x1a6, 0x5, 0x58, 0x2d, 0x2, 0x1a6, 
    0x4b, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x1a8, 0x7, 0x16, 0x2, 0x2, 0x1a8, 
    0x1a9, 0x5, 0x56, 0x2c, 0x2, 0x1a9, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x1aa, 
    0x1ab, 0x7, 0x13, 0x2, 0x2, 0x1ab, 0x1ac, 0x7, 0x4a, 0x2, 0x2, 0x1ac, 
    0x1b1, 0x5, 0x54, 0x2b, 0x2, 0x1ad, 0x1ae, 0x7, 0xd, 0x2, 0x2, 0x1ae, 
    0x1b0, 0x5, 0x54, 0x2b, 0x2, 0x1af, 0x1ad, 0x3, 0x2, 0x2, 0x2, 0x1b0, 
    0x1b3, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x1af, 0x3, 0x2, 0x2, 0x2, 0x1b1, 
    0x1b2, 0x3, 0x2, 0x2, 0x2, 0x1b2, 0x1b4, 0x3, 0x2, 0x2, 0x2, 0x1b3, 
    0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b4, 0x1b5, 0x7, 0x4b, 0x2, 0x2, 0x1b5, 
    0x4f, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b7, 0x7, 0x15, 0x2, 0x2, 0x1b7, 
    0x1b8, 0x7, 0x4a, 0x2, 0x2, 0x1b8, 0x1bd, 0x5, 0x58, 0x2d, 0x2, 0x1b9, 
    0x1ba, 0x7, 0xd, 0x2, 0x2, 0x1ba, 0x1bc, 0x5, 0x58, 0x2d, 0x2, 0x1bb, 
    0x1b9, 0x3, 0x2, 0x2, 0x2, 0x1bc, 0x1bf, 0x3, 0x2, 0x2, 0x2, 0x1bd, 
    0x1bb, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1be, 0x3, 0x2, 0x2, 0x2, 0x1be, 
    0x1c0, 0x3, 0x2, 0x2, 0x2, 0x1bf, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x1c0, 
    0x1c1, 0x7, 0x4b, 0x2, 0x2, 0x1c1, 0x51, 0x3, 0x2, 0x2, 0x2, 0x1c2, 
    0x1c3, 0x7, 0x17, 0x2, 0x2, 0x1c3, 0x1c4, 0x7, 0x4a, 0x2, 0x2, 0x1c4, 
    0x1c9, 0x5, 0x56, 0x2c, 0x2, 0x1c5, 0x1c6, 0x7, 0xd, 0x2, 0x2, 0x1c6, 
    0x1c8, 0x5, 0x56, 0x2c, 0x2, 0x1c7, 0x1c5, 0x3, 0x2, 0x2, 0x2, 0x1c8, 
    0x1cb, 0x3, 0x2, 0x2, 0x2, 0x1c9, 0x1c7, 0x3, 0x2, 0x2, 0x2, 0x1c9, 
    0x1ca, 0x3, 0x2, 0x2, 0x2, 0x1ca, 0x1cc, 0x3, 0x2, 0x2, 0x2, 0x1cb, 
    0x1c9, 0x3, 0x2, 0x2, 0x2, 0x1cc, 0x1cd, 0x7, 0x4b, 0x2, 0x2, 0x1cd, 
    0x53, 0x3, 0x2, 0x2, 0x2, 0x1ce, 0x1d4, 0x5, 0x5a, 0x2e, 0x2, 0x1cf, 
    0x1d0, 0x7, 0x4a, 0x2, 0x2, 0x1d0, 0x1d1, 0x5, 0x5a, 0x2e, 0x2, 0x1d1, 
    0x1d2, 0x7, 0x4b, 0x2, 0x2, 0x1d2, 0x1d4, 0x3, 0x2, 0x2, 0x2, 0x1d3, 
    0x1ce, 0x3, 0x2, 0x2, 0x2, 0x1d3, 0x1cf, 0x3, 0x2, 0x2, 0x2, 0x1d4, 
    0x55, 0x3, 0x2, 0x2, 0x2, 0x1d5, 0x1d6, 0x7, 0x4a, 0x2, 0x2, 0x1d6, 
    0x1db, 0x5, 0x58, 0x2d, 0x2, 0x1d7, 0x1d8, 0x7, 0xd, 0x2, 0x2, 0x1d8, 
    0x1da, 0x5, 0x58, 0x2d, 0x2, 0x1d9, 0x1d7, 0x3, 0x2, 0x2, 0x2, 0x1da, 
    0x1dd, 0x3, 0x2, 0x2, 0x2, 0x1db, 0x1d9, 0x3, 0x2, 0x2, 0x2, 0x1db, 
    0x1dc, 0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1de, 0x3, 0x2, 0x2, 0x2, 0x1dd, 
    0x1db, 0x3, 0x2, 0x2, 0x2, 0x1de, 0x1df, 0x7, 0x4b, 0x2, 0x2, 0x1df, 
    0x57, 0x3, 0x2, 0x2, 0x2, 0x1e0, 0x1e1, 0x7, 0x4a, 0x2, 0x2, 0x1e1, 
    0x1e6, 0x5, 0x5a, 0x2e, 0x2, 0x1e2, 0x1e3, 0x7, 0xd, 0x2, 0x2, 0x1e3, 
    0x1e5, 0x5, 0x5a, 0x2e, 0x2, 0x1e4, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x1e5, 
    0x1e8, 0x3, 0x2, 0x2, 0x2, 0x1e6, 0x1e4, 0x3, 0x2, 0x2, 0x2, 0x1e6, 
    0x1e7, 0x3, 0x2, 0x2, 0x2, 0x1e7, 0x1e9, 0x3, 0x2, 0x2, 0x2, 0x1e8, 
    0x1e6, 0x3, 0x2, 0x2, 0x2, 0x1e9, 0x1ea, 0x7, 0x4b, 0x2, 0x2, 0x1ea, 
    0x59, 0x3, 0x2, 0x2, 0x2, 0x1eb, 0x1ec, 0x9, 0x8, 0x2, 0x2, 0x1ec, 0x1ed, 
    0x9, 0x8, 0x2, 0x2, 0x1ed, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x23, 0x5f, 0x69, 
    0x6e, 0x76, 0x7f, 0x88, 0x8c, 0x90, 0x94, 0x98, 0x9c, 0xba, 0xc5, 0xcb, 
    0xd4, 0xd9, 0xe0, 0xe8, 0xf0, 0xfa, 0x101, 0x107, 0x121, 0x172, 0x192, 
    0x194, 0x19d, 0x1b1, 0x1bd, 0x1c9, 0x1d3, 0x1db, 0x1e6, 
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


// Generated from C:/Users/mstano/GPU-DB/dropdbase/GpuSqlParser\GpuSqlParser.g4 by ANTLR 4.7.2


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
    setState(123);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << GpuSqlParser::INSERTINTO)
      | (1ULL << GpuSqlParser::CREATEDB)
      | (1ULL << GpuSqlParser::DROPDB)
      | (1ULL << GpuSqlParser::CREATETABLE)
      | (1ULL << GpuSqlParser::DROPTABLE)
      | (1ULL << GpuSqlParser::ALTERTABLE)
      | (1ULL << GpuSqlParser::CREATEINDEX)
      | (1ULL << GpuSqlParser::SELECT)
      | (1ULL << GpuSqlParser::SHOWDB)
      | (1ULL << GpuSqlParser::SHOWTB)
      | (1ULL << GpuSqlParser::SHOWCL))) != 0)) {
      setState(120);
      statement();
      setState(125);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(126);
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

GpuSqlParser::SqlDropDbContext* GpuSqlParser::StatementContext::sqlDropDb() {
  return getRuleContext<GpuSqlParser::SqlDropDbContext>(0);
}

GpuSqlParser::SqlCreateTableContext* GpuSqlParser::StatementContext::sqlCreateTable() {
  return getRuleContext<GpuSqlParser::SqlCreateTableContext>(0);
}

GpuSqlParser::SqlDropTableContext* GpuSqlParser::StatementContext::sqlDropTable() {
  return getRuleContext<GpuSqlParser::SqlDropTableContext>(0);
}

GpuSqlParser::SqlAlterTableContext* GpuSqlParser::StatementContext::sqlAlterTable() {
  return getRuleContext<GpuSqlParser::SqlAlterTableContext>(0);
}

GpuSqlParser::SqlCreateIndexContext* GpuSqlParser::StatementContext::sqlCreateIndex() {
  return getRuleContext<GpuSqlParser::SqlCreateIndexContext>(0);
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
    setState(137);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SELECT: {
        enterOuterAlt(_localctx, 1);
        setState(128);
        sqlSelect();
        break;
      }

      case GpuSqlParser::CREATEDB: {
        enterOuterAlt(_localctx, 2);
        setState(129);
        sqlCreateDb();
        break;
      }

      case GpuSqlParser::DROPDB: {
        enterOuterAlt(_localctx, 3);
        setState(130);
        sqlDropDb();
        break;
      }

      case GpuSqlParser::CREATETABLE: {
        enterOuterAlt(_localctx, 4);
        setState(131);
        sqlCreateTable();
        break;
      }

      case GpuSqlParser::DROPTABLE: {
        enterOuterAlt(_localctx, 5);
        setState(132);
        sqlDropTable();
        break;
      }

      case GpuSqlParser::ALTERTABLE: {
        enterOuterAlt(_localctx, 6);
        setState(133);
        sqlAlterTable();
        break;
      }

      case GpuSqlParser::CREATEINDEX: {
        enterOuterAlt(_localctx, 7);
        setState(134);
        sqlCreateIndex();
        break;
      }

      case GpuSqlParser::INSERTINTO: {
        enterOuterAlt(_localctx, 8);
        setState(135);
        sqlInsertInto();
        break;
      }

      case GpuSqlParser::SHOWDB:
      case GpuSqlParser::SHOWTB:
      case GpuSqlParser::SHOWCL: {
        enterOuterAlt(_localctx, 9);
        setState(136);
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
    setState(142);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SHOWDB: {
        setState(139);
        showDatabases();
        break;
      }

      case GpuSqlParser::SHOWTB: {
        setState(140);
        showTables();
        break;
      }

      case GpuSqlParser::SHOWCL: {
        setState(141);
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
    setState(144);
    match(GpuSqlParser::SHOWDB);
    setState(145);
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
    setState(147);
    match(GpuSqlParser::SHOWTB);
    setState(150);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(148);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(149);
      database();
    }
    setState(152);
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
    setState(154);
    match(GpuSqlParser::SHOWCL);
    setState(155);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(156);
    table();
    setState(159);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(157);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(158);
      database();
    }
    setState(161);
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
    setState(163);
    match(GpuSqlParser::SELECT);
    setState(164);
    selectColumns();
    setState(165);
    match(GpuSqlParser::FROM);
    setState(166);
    fromTables();
    setState(168);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::JOIN) {
      setState(167);
      joinClauses();
    }
    setState(172);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::WHERE) {
      setState(170);
      match(GpuSqlParser::WHERE);
      setState(171);
      whereClause();
    }
    setState(176);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::GROUPBY) {
      setState(174);
      match(GpuSqlParser::GROUPBY);
      setState(175);
      groupByColumns();
    }
    setState(180);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::ORDERBY) {
      setState(178);
      match(GpuSqlParser::ORDERBY);
      setState(179);
      orderByColumns();
    }
    setState(184);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::LIMIT) {
      setState(182);
      match(GpuSqlParser::LIMIT);
      setState(183);
      limit();
    }
    setState(188);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::OFFSET) {
      setState(186);
      match(GpuSqlParser::OFFSET);
      setState(187);
      offset();
    }
    setState(190);
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

GpuSqlParser::BlockSizeContext* GpuSqlParser::SqlCreateDbContext::blockSize() {
  return getRuleContext<GpuSqlParser::BlockSizeContext>(0);
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
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(192);
    match(GpuSqlParser::CREATEDB);
    setState(193);
    database();
    setState(195);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::INTLIT) {
      setState(194);
      blockSize();
    }
    setState(197);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlDropDbContext ------------------------------------------------------------------

GpuSqlParser::SqlDropDbContext::SqlDropDbContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlDropDbContext::DROPDB() {
  return getToken(GpuSqlParser::DROPDB, 0);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::SqlDropDbContext::database() {
  return getRuleContext<GpuSqlParser::DatabaseContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlDropDbContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::SqlDropDbContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlDropDb;
}

void GpuSqlParser::SqlDropDbContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlDropDb(this);
}

void GpuSqlParser::SqlDropDbContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlDropDb(this);
}

GpuSqlParser::SqlDropDbContext* GpuSqlParser::sqlDropDb() {
  SqlDropDbContext *_localctx = _tracker.createInstance<SqlDropDbContext>(_ctx, getState());
  enterRule(_localctx, 16, GpuSqlParser::RuleSqlDropDb);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(199);
    match(GpuSqlParser::DROPDB);
    setState(200);
    database();
    setState(201);
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

GpuSqlParser::NewTableEntriesContext* GpuSqlParser::SqlCreateTableContext::newTableEntries() {
  return getRuleContext<GpuSqlParser::NewTableEntriesContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateTableContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateTableContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
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
  enterRule(_localctx, 18, GpuSqlParser::RuleSqlCreateTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(203);
    match(GpuSqlParser::CREATETABLE);
    setState(204);
    table();
    setState(205);
    match(GpuSqlParser::LPAREN);
    setState(206);
    newTableEntries();
    setState(207);
    match(GpuSqlParser::RPAREN);
    setState(208);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlDropTableContext ------------------------------------------------------------------

GpuSqlParser::SqlDropTableContext::SqlDropTableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlDropTableContext::DROPTABLE() {
  return getToken(GpuSqlParser::DROPTABLE, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::SqlDropTableContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlDropTableContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::SqlDropTableContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlDropTable;
}

void GpuSqlParser::SqlDropTableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlDropTable(this);
}

void GpuSqlParser::SqlDropTableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlDropTable(this);
}

GpuSqlParser::SqlDropTableContext* GpuSqlParser::sqlDropTable() {
  SqlDropTableContext *_localctx = _tracker.createInstance<SqlDropTableContext>(_ctx, getState());
  enterRule(_localctx, 20, GpuSqlParser::RuleSqlDropTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(210);
    match(GpuSqlParser::DROPTABLE);
    setState(211);
    table();
    setState(212);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlAlterTableContext ------------------------------------------------------------------

GpuSqlParser::SqlAlterTableContext::SqlAlterTableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlAlterTableContext::ALTERTABLE() {
  return getToken(GpuSqlParser::ALTERTABLE, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::SqlAlterTableContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

GpuSqlParser::AlterTableEntriesContext* GpuSqlParser::SqlAlterTableContext::alterTableEntries() {
  return getRuleContext<GpuSqlParser::AlterTableEntriesContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlAlterTableContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::SqlAlterTableContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlAlterTable;
}

void GpuSqlParser::SqlAlterTableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlAlterTable(this);
}

void GpuSqlParser::SqlAlterTableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlAlterTable(this);
}

GpuSqlParser::SqlAlterTableContext* GpuSqlParser::sqlAlterTable() {
  SqlAlterTableContext *_localctx = _tracker.createInstance<SqlAlterTableContext>(_ctx, getState());
  enterRule(_localctx, 22, GpuSqlParser::RuleSqlAlterTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(214);
    match(GpuSqlParser::ALTERTABLE);
    setState(215);
    table();
    setState(216);
    alterTableEntries();
    setState(217);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlCreateIndexContext ------------------------------------------------------------------

GpuSqlParser::SqlCreateIndexContext::SqlCreateIndexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlCreateIndexContext::CREATEINDEX() {
  return getToken(GpuSqlParser::CREATEINDEX, 0);
}

GpuSqlParser::IndexNameContext* GpuSqlParser::SqlCreateIndexContext::indexName() {
  return getRuleContext<GpuSqlParser::IndexNameContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateIndexContext::ON() {
  return getToken(GpuSqlParser::ON, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::SqlCreateIndexContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateIndexContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::IndexColumnsContext* GpuSqlParser::SqlCreateIndexContext::indexColumns() {
  return getRuleContext<GpuSqlParser::IndexColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateIndexContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateIndexContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::SqlCreateIndexContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlCreateIndex;
}

void GpuSqlParser::SqlCreateIndexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlCreateIndex(this);
}

void GpuSqlParser::SqlCreateIndexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlCreateIndex(this);
}

GpuSqlParser::SqlCreateIndexContext* GpuSqlParser::sqlCreateIndex() {
  SqlCreateIndexContext *_localctx = _tracker.createInstance<SqlCreateIndexContext>(_ctx, getState());
  enterRule(_localctx, 24, GpuSqlParser::RuleSqlCreateIndex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(219);
    match(GpuSqlParser::CREATEINDEX);
    setState(220);
    indexName();
    setState(221);
    match(GpuSqlParser::ON);
    setState(222);
    table();
    setState(223);
    match(GpuSqlParser::LPAREN);
    setState(224);
    indexColumns();
    setState(225);
    match(GpuSqlParser::RPAREN);
    setState(226);
    match(GpuSqlParser::SEMICOL);
   
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
  enterRule(_localctx, 26, GpuSqlParser::RuleSqlInsertInto);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(228);
    match(GpuSqlParser::INSERTINTO);
    setState(229);
    table();
    setState(230);
    match(GpuSqlParser::LPAREN);
    setState(231);
    insertIntoColumns();
    setState(232);
    match(GpuSqlParser::RPAREN);
    setState(233);
    match(GpuSqlParser::VALUES);
    setState(234);
    match(GpuSqlParser::LPAREN);
    setState(235);
    insertIntoValues();
    setState(236);
    match(GpuSqlParser::RPAREN);
    setState(237);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NewTableEntriesContext ------------------------------------------------------------------

GpuSqlParser::NewTableEntriesContext::NewTableEntriesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::NewTableEntryContext *> GpuSqlParser::NewTableEntriesContext::newTableEntry() {
  return getRuleContexts<GpuSqlParser::NewTableEntryContext>();
}

GpuSqlParser::NewTableEntryContext* GpuSqlParser::NewTableEntriesContext::newTableEntry(size_t i) {
  return getRuleContext<GpuSqlParser::NewTableEntryContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::NewTableEntriesContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::NewTableEntriesContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::NewTableEntriesContext::getRuleIndex() const {
  return GpuSqlParser::RuleNewTableEntries;
}

void GpuSqlParser::NewTableEntriesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNewTableEntries(this);
}

void GpuSqlParser::NewTableEntriesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNewTableEntries(this);
}

GpuSqlParser::NewTableEntriesContext* GpuSqlParser::newTableEntries() {
  NewTableEntriesContext *_localctx = _tracker.createInstance<NewTableEntriesContext>(_ctx, getState());
  enterRule(_localctx, 28, GpuSqlParser::RuleNewTableEntries);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(239);
    newTableEntry();
    setState(244);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(240);
      match(GpuSqlParser::COMMA);
      setState(241);
      newTableEntry();
      setState(246);
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

//----------------- NewTableEntryContext ------------------------------------------------------------------

GpuSqlParser::NewTableEntryContext::NewTableEntryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::NewTableColumnContext* GpuSqlParser::NewTableEntryContext::newTableColumn() {
  return getRuleContext<GpuSqlParser::NewTableColumnContext>(0);
}

GpuSqlParser::NewTableIndexContext* GpuSqlParser::NewTableEntryContext::newTableIndex() {
  return getRuleContext<GpuSqlParser::NewTableIndexContext>(0);
}


size_t GpuSqlParser::NewTableEntryContext::getRuleIndex() const {
  return GpuSqlParser::RuleNewTableEntry;
}

void GpuSqlParser::NewTableEntryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNewTableEntry(this);
}

void GpuSqlParser::NewTableEntryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNewTableEntry(this);
}

GpuSqlParser::NewTableEntryContext* GpuSqlParser::newTableEntry() {
  NewTableEntryContext *_localctx = _tracker.createInstance<NewTableEntryContext>(_ctx, getState());
  enterRule(_localctx, 30, GpuSqlParser::RuleNewTableEntry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(249);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::DELIMID:
      case GpuSqlParser::ID: {
        setState(247);
        newTableColumn();
        break;
      }

      case GpuSqlParser::INDEX: {
        setState(248);
        newTableIndex();
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

//----------------- AlterTableEntriesContext ------------------------------------------------------------------

GpuSqlParser::AlterTableEntriesContext::AlterTableEntriesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::AlterTableEntryContext *> GpuSqlParser::AlterTableEntriesContext::alterTableEntry() {
  return getRuleContexts<GpuSqlParser::AlterTableEntryContext>();
}

GpuSqlParser::AlterTableEntryContext* GpuSqlParser::AlterTableEntriesContext::alterTableEntry(size_t i) {
  return getRuleContext<GpuSqlParser::AlterTableEntryContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::AlterTableEntriesContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::AlterTableEntriesContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::AlterTableEntriesContext::getRuleIndex() const {
  return GpuSqlParser::RuleAlterTableEntries;
}

void GpuSqlParser::AlterTableEntriesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAlterTableEntries(this);
}

void GpuSqlParser::AlterTableEntriesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAlterTableEntries(this);
}

GpuSqlParser::AlterTableEntriesContext* GpuSqlParser::alterTableEntries() {
  AlterTableEntriesContext *_localctx = _tracker.createInstance<AlterTableEntriesContext>(_ctx, getState());
  enterRule(_localctx, 32, GpuSqlParser::RuleAlterTableEntries);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(251);
    alterTableEntry();
    setState(256);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(252);
      match(GpuSqlParser::COMMA);
      setState(253);
      alterTableEntry();
      setState(258);
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

//----------------- AlterTableEntryContext ------------------------------------------------------------------

GpuSqlParser::AlterTableEntryContext::AlterTableEntryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::AddColumnContext* GpuSqlParser::AlterTableEntryContext::addColumn() {
  return getRuleContext<GpuSqlParser::AddColumnContext>(0);
}

GpuSqlParser::DropColumnContext* GpuSqlParser::AlterTableEntryContext::dropColumn() {
  return getRuleContext<GpuSqlParser::DropColumnContext>(0);
}

GpuSqlParser::AlterColumnContext* GpuSqlParser::AlterTableEntryContext::alterColumn() {
  return getRuleContext<GpuSqlParser::AlterColumnContext>(0);
}


size_t GpuSqlParser::AlterTableEntryContext::getRuleIndex() const {
  return GpuSqlParser::RuleAlterTableEntry;
}

void GpuSqlParser::AlterTableEntryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAlterTableEntry(this);
}

void GpuSqlParser::AlterTableEntryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAlterTableEntry(this);
}

GpuSqlParser::AlterTableEntryContext* GpuSqlParser::alterTableEntry() {
  AlterTableEntryContext *_localctx = _tracker.createInstance<AlterTableEntryContext>(_ctx, getState());
  enterRule(_localctx, 34, GpuSqlParser::RuleAlterTableEntry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(262);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::ADD: {
        setState(259);
        addColumn();
        break;
      }

      case GpuSqlParser::DROPCOLUMN: {
        setState(260);
        dropColumn();
        break;
      }

      case GpuSqlParser::ALTERCOLUMN: {
        setState(261);
        alterColumn();
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

//----------------- AddColumnContext ------------------------------------------------------------------

GpuSqlParser::AddColumnContext::AddColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::AddColumnContext::ADD() {
  return getToken(GpuSqlParser::ADD, 0);
}

GpuSqlParser::ColumnContext* GpuSqlParser::AddColumnContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
}

tree::TerminalNode* GpuSqlParser::AddColumnContext::DATATYPE() {
  return getToken(GpuSqlParser::DATATYPE, 0);
}


size_t GpuSqlParser::AddColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleAddColumn;
}

void GpuSqlParser::AddColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAddColumn(this);
}

void GpuSqlParser::AddColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAddColumn(this);
}

GpuSqlParser::AddColumnContext* GpuSqlParser::addColumn() {
  AddColumnContext *_localctx = _tracker.createInstance<AddColumnContext>(_ctx, getState());
  enterRule(_localctx, 36, GpuSqlParser::RuleAddColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(264);
    match(GpuSqlParser::ADD);
    setState(265);
    column();
    setState(266);
    match(GpuSqlParser::DATATYPE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DropColumnContext ------------------------------------------------------------------

GpuSqlParser::DropColumnContext::DropColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::DropColumnContext::DROPCOLUMN() {
  return getToken(GpuSqlParser::DROPCOLUMN, 0);
}

GpuSqlParser::ColumnContext* GpuSqlParser::DropColumnContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
}


size_t GpuSqlParser::DropColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleDropColumn;
}

void GpuSqlParser::DropColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDropColumn(this);
}

void GpuSqlParser::DropColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDropColumn(this);
}

GpuSqlParser::DropColumnContext* GpuSqlParser::dropColumn() {
  DropColumnContext *_localctx = _tracker.createInstance<DropColumnContext>(_ctx, getState());
  enterRule(_localctx, 38, GpuSqlParser::RuleDropColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(268);
    match(GpuSqlParser::DROPCOLUMN);
    setState(269);
    column();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AlterColumnContext ------------------------------------------------------------------

GpuSqlParser::AlterColumnContext::AlterColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::AlterColumnContext::ALTERCOLUMN() {
  return getToken(GpuSqlParser::ALTERCOLUMN, 0);
}

GpuSqlParser::ColumnContext* GpuSqlParser::AlterColumnContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
}

tree::TerminalNode* GpuSqlParser::AlterColumnContext::DATATYPE() {
  return getToken(GpuSqlParser::DATATYPE, 0);
}


size_t GpuSqlParser::AlterColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleAlterColumn;
}

void GpuSqlParser::AlterColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAlterColumn(this);
}

void GpuSqlParser::AlterColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAlterColumn(this);
}

GpuSqlParser::AlterColumnContext* GpuSqlParser::alterColumn() {
  AlterColumnContext *_localctx = _tracker.createInstance<AlterColumnContext>(_ctx, getState());
  enterRule(_localctx, 40, GpuSqlParser::RuleAlterColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(271);
    match(GpuSqlParser::ALTERCOLUMN);
    setState(272);
    column();
    setState(273);
    match(GpuSqlParser::DATATYPE);
   
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

GpuSqlParser::ColumnContext* GpuSqlParser::NewTableColumnContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
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
  enterRule(_localctx, 42, GpuSqlParser::RuleNewTableColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(275);
    column();
    setState(276);
    match(GpuSqlParser::DATATYPE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NewTableIndexContext ------------------------------------------------------------------

GpuSqlParser::NewTableIndexContext::NewTableIndexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::NewTableIndexContext::INDEX() {
  return getToken(GpuSqlParser::INDEX, 0);
}

GpuSqlParser::IndexNameContext* GpuSqlParser::NewTableIndexContext::indexName() {
  return getRuleContext<GpuSqlParser::IndexNameContext>(0);
}

tree::TerminalNode* GpuSqlParser::NewTableIndexContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::IndexColumnsContext* GpuSqlParser::NewTableIndexContext::indexColumns() {
  return getRuleContext<GpuSqlParser::IndexColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::NewTableIndexContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}


size_t GpuSqlParser::NewTableIndexContext::getRuleIndex() const {
  return GpuSqlParser::RuleNewTableIndex;
}

void GpuSqlParser::NewTableIndexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNewTableIndex(this);
}

void GpuSqlParser::NewTableIndexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNewTableIndex(this);
}

GpuSqlParser::NewTableIndexContext* GpuSqlParser::newTableIndex() {
  NewTableIndexContext *_localctx = _tracker.createInstance<NewTableIndexContext>(_ctx, getState());
  enterRule(_localctx, 44, GpuSqlParser::RuleNewTableIndex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(278);
    match(GpuSqlParser::INDEX);
    setState(279);
    indexName();
    setState(280);
    match(GpuSqlParser::LPAREN);
    setState(281);
    indexColumns();
    setState(282);
    match(GpuSqlParser::RPAREN);
   
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
  enterRule(_localctx, 46, GpuSqlParser::RuleSelectColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(284);
    selectColumn();
    setState(289);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(285);
      match(GpuSqlParser::COMMA);
      setState(286);
      selectColumn();
      setState(291);
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
  enterRule(_localctx, 48, GpuSqlParser::RuleSelectColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(292);
    expression(0);
    setState(295);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(293);
      match(GpuSqlParser::AS);
      setState(294);
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
  enterRule(_localctx, 50, GpuSqlParser::RuleWhereClause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(297);
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
  enterRule(_localctx, 52, GpuSqlParser::RuleOrderByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(299);
    orderByColumn();
    setState(304);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(300);
      match(GpuSqlParser::COMMA);
      setState(301);
      orderByColumn();
      setState(306);
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
  enterRule(_localctx, 54, GpuSqlParser::RuleOrderByColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(307);
    columnId();
    setState(309);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::DIR) {
      setState(308);
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
  enterRule(_localctx, 56, GpuSqlParser::RuleInsertIntoValues);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(311);
    columnValue();
    setState(316);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(312);
      match(GpuSqlParser::COMMA);
      setState(313);
      columnValue();
      setState(318);
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
  enterRule(_localctx, 58, GpuSqlParser::RuleInsertIntoColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(319);
    columnId();
    setState(324);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(320);
      match(GpuSqlParser::COMMA);
      setState(321);
      columnId();
      setState(326);
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

//----------------- IndexColumnsContext ------------------------------------------------------------------

GpuSqlParser::IndexColumnsContext::IndexColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::ColumnContext *> GpuSqlParser::IndexColumnsContext::column() {
  return getRuleContexts<GpuSqlParser::ColumnContext>();
}

GpuSqlParser::ColumnContext* GpuSqlParser::IndexColumnsContext::column(size_t i) {
  return getRuleContext<GpuSqlParser::ColumnContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::IndexColumnsContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::IndexColumnsContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::IndexColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleIndexColumns;
}

void GpuSqlParser::IndexColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexColumns(this);
}

void GpuSqlParser::IndexColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexColumns(this);
}

GpuSqlParser::IndexColumnsContext* GpuSqlParser::indexColumns() {
  IndexColumnsContext *_localctx = _tracker.createInstance<IndexColumnsContext>(_ctx, getState());
  enterRule(_localctx, 60, GpuSqlParser::RuleIndexColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(327);
    column();
    setState(332);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(328);
      match(GpuSqlParser::COMMA);
      setState(329);
      column();
      setState(334);
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
  enterRule(_localctx, 62, GpuSqlParser::RuleGroupByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(335);
    groupByColumn();
    setState(340);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(336);
      match(GpuSqlParser::COMMA);
      setState(337);
      groupByColumn();
      setState(342);
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
  enterRule(_localctx, 64, GpuSqlParser::RuleGroupByColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(343);
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
  enterRule(_localctx, 66, GpuSqlParser::RuleColumnId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(350);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(345);
      column();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(346);
      table();
      setState(347);
      match(GpuSqlParser::DOT);
      setState(348);
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

std::vector<GpuSqlParser::FromTableContext *> GpuSqlParser::FromTablesContext::fromTable() {
  return getRuleContexts<GpuSqlParser::FromTableContext>();
}

GpuSqlParser::FromTableContext* GpuSqlParser::FromTablesContext::fromTable(size_t i) {
  return getRuleContext<GpuSqlParser::FromTableContext>(i);
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
  enterRule(_localctx, 68, GpuSqlParser::RuleFromTables);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(352);
    fromTable();
    setState(357);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(353);
      match(GpuSqlParser::COMMA);
      setState(354);
      fromTable();
      setState(359);
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
  enterRule(_localctx, 70, GpuSqlParser::RuleJoinClauses);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(361); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(360);
      joinClause();
      setState(363); 
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
  enterRule(_localctx, 72, GpuSqlParser::RuleJoinClause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(365);
    match(GpuSqlParser::JOIN);
    setState(366);
    joinTable();
    setState(367);
    match(GpuSqlParser::ON);
    setState(368);
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

tree::TerminalNode* GpuSqlParser::JoinTableContext::AS() {
  return getToken(GpuSqlParser::AS, 0);
}

GpuSqlParser::AliasContext* GpuSqlParser::JoinTableContext::alias() {
  return getRuleContext<GpuSqlParser::AliasContext>(0);
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
  enterRule(_localctx, 74, GpuSqlParser::RuleJoinTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(370);
    table();
    setState(373);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(371);
      match(GpuSqlParser::AS);
      setState(372);
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

//----------------- FromTableContext ------------------------------------------------------------------

GpuSqlParser::FromTableContext::FromTableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::TableContext* GpuSqlParser::FromTableContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

tree::TerminalNode* GpuSqlParser::FromTableContext::AS() {
  return getToken(GpuSqlParser::AS, 0);
}

GpuSqlParser::AliasContext* GpuSqlParser::FromTableContext::alias() {
  return getRuleContext<GpuSqlParser::AliasContext>(0);
}


size_t GpuSqlParser::FromTableContext::getRuleIndex() const {
  return GpuSqlParser::RuleFromTable;
}

void GpuSqlParser::FromTableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFromTable(this);
}

void GpuSqlParser::FromTableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFromTable(this);
}

GpuSqlParser::FromTableContext* GpuSqlParser::fromTable() {
  FromTableContext *_localctx = _tracker.createInstance<FromTableContext>(_ctx, getState());
  enterRule(_localctx, 76, GpuSqlParser::RuleFromTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(375);
    table();
    setState(378);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(376);
      match(GpuSqlParser::AS);
      setState(377);
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

//----------------- TableContext ------------------------------------------------------------------

GpuSqlParser::TableContext::TableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::TableContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}

tree::TerminalNode* GpuSqlParser::TableContext::DELIMID() {
  return getToken(GpuSqlParser::DELIMID, 0);
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
  enterRule(_localctx, 78, GpuSqlParser::RuleTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(380);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::DELIMID || _la == GpuSqlParser::ID)) {
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

//----------------- ColumnContext ------------------------------------------------------------------

GpuSqlParser::ColumnContext::ColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ColumnContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}

tree::TerminalNode* GpuSqlParser::ColumnContext::DELIMID() {
  return getToken(GpuSqlParser::DELIMID, 0);
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
  enterRule(_localctx, 80, GpuSqlParser::RuleColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(382);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::DELIMID || _la == GpuSqlParser::ID)) {
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

//----------------- DatabaseContext ------------------------------------------------------------------

GpuSqlParser::DatabaseContext::DatabaseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::DatabaseContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}

tree::TerminalNode* GpuSqlParser::DatabaseContext::DELIMID() {
  return getToken(GpuSqlParser::DELIMID, 0);
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
  enterRule(_localctx, 82, GpuSqlParser::RuleDatabase);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(384);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::DELIMID || _la == GpuSqlParser::ID)) {
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

//----------------- AliasContext ------------------------------------------------------------------

GpuSqlParser::AliasContext::AliasContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::AliasContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}

tree::TerminalNode* GpuSqlParser::AliasContext::DELIMID() {
  return getToken(GpuSqlParser::DELIMID, 0);
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
  enterRule(_localctx, 84, GpuSqlParser::RuleAlias);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(386);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::DELIMID || _la == GpuSqlParser::ID)) {
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

//----------------- IndexNameContext ------------------------------------------------------------------

GpuSqlParser::IndexNameContext::IndexNameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::IndexNameContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}

tree::TerminalNode* GpuSqlParser::IndexNameContext::DELIMID() {
  return getToken(GpuSqlParser::DELIMID, 0);
}


size_t GpuSqlParser::IndexNameContext::getRuleIndex() const {
  return GpuSqlParser::RuleIndexName;
}

void GpuSqlParser::IndexNameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndexName(this);
}

void GpuSqlParser::IndexNameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndexName(this);
}

GpuSqlParser::IndexNameContext* GpuSqlParser::indexName() {
  IndexNameContext *_localctx = _tracker.createInstance<IndexNameContext>(_ctx, getState());
  enterRule(_localctx, 86, GpuSqlParser::RuleIndexName);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(388);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::DELIMID || _la == GpuSqlParser::ID)) {
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
  enterRule(_localctx, 88, GpuSqlParser::RuleLimit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(390);
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
  enterRule(_localctx, 90, GpuSqlParser::RuleOffset);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(392);
    match(GpuSqlParser::INTLIT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockSizeContext ------------------------------------------------------------------

GpuSqlParser::BlockSizeContext::BlockSizeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::BlockSizeContext::INTLIT() {
  return getToken(GpuSqlParser::INTLIT, 0);
}


size_t GpuSqlParser::BlockSizeContext::getRuleIndex() const {
  return GpuSqlParser::RuleBlockSize;
}

void GpuSqlParser::BlockSizeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBlockSize(this);
}

void GpuSqlParser::BlockSizeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBlockSize(this);
}

GpuSqlParser::BlockSizeContext* GpuSqlParser::blockSize() {
  BlockSizeContext *_localctx = _tracker.createInstance<BlockSizeContext>(_ctx, getState());
  enterRule(_localctx, 92, GpuSqlParser::RuleBlockSize);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(394);
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

tree::TerminalNode* GpuSqlParser::ColumnValueContext::STRING() {
  return getToken(GpuSqlParser::STRING, 0);
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
  enterRule(_localctx, 94, GpuSqlParser::RuleColumnValue);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(400);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::INTLIT: {
        setState(396);
        match(GpuSqlParser::INTLIT);
        break;
      }

      case GpuSqlParser::FLOATLIT: {
        setState(397);
        match(GpuSqlParser::FLOATLIT);
        break;
      }

      case GpuSqlParser::POINT:
      case GpuSqlParser::MULTIPOINT:
      case GpuSqlParser::LINESTRING:
      case GpuSqlParser::MULTILINESTRING:
      case GpuSqlParser::POLYGON:
      case GpuSqlParser::MULTIPOLYGON: {
        setState(398);
        geometry();
        break;
      }

      case GpuSqlParser::STRING: {
        setState(399);
        match(GpuSqlParser::STRING);
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
//----------------- NowLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::NowLiteralContext::NOW() {
  return getToken(GpuSqlParser::NOW, 0);
}

GpuSqlParser::NowLiteralContext::NowLiteralContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::NowLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNowLiteral(this);
}
void GpuSqlParser::NowLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNowLiteral(this);
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

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::ATAN2() {
  return getToken(GpuSqlParser::ATAN2, 0);
}

std::vector<GpuSqlParser::ExpressionContext *> GpuSqlParser::BinaryOperationContext::expression() {
  return getRuleContexts<GpuSqlParser::ExpressionContext>();
}

GpuSqlParser::ExpressionContext* GpuSqlParser::BinaryOperationContext::expression(size_t i) {
  return getRuleContext<GpuSqlParser::ExpressionContext>(i);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::LOG() {
  return getToken(GpuSqlParser::LOG, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::POW() {
  return getToken(GpuSqlParser::POW, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::ROOT() {
  return getToken(GpuSqlParser::ROOT, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::POINT() {
  return getToken(GpuSqlParser::POINT, 0);
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

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::CONCAT() {
  return getToken(GpuSqlParser::CONCAT, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::LEFT() {
  return getToken(GpuSqlParser::LEFT, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::RIGHT() {
  return getToken(GpuSqlParser::RIGHT, 0);
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

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::MODULO() {
  return getToken(GpuSqlParser::MODULO, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::XOR() {
  return getToken(GpuSqlParser::XOR, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::BIT_AND() {
  return getToken(GpuSqlParser::BIT_AND, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::BIT_OR() {
  return getToken(GpuSqlParser::BIT_OR, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::L_SHIFT() {
  return getToken(GpuSqlParser::L_SHIFT, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::R_SHIFT() {
  return getToken(GpuSqlParser::R_SHIFT, 0);
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

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::NOTEQUALS_GT_LT() {
  return getToken(GpuSqlParser::NOTEQUALS_GT_LT, 0);
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

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::ABS() {
  return getToken(GpuSqlParser::ABS, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::SIN() {
  return getToken(GpuSqlParser::SIN, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::COS() {
  return getToken(GpuSqlParser::COS, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::TAN() {
  return getToken(GpuSqlParser::TAN, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::COT() {
  return getToken(GpuSqlParser::COT, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::ASIN() {
  return getToken(GpuSqlParser::ASIN, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::ACOS() {
  return getToken(GpuSqlParser::ACOS, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::ATAN() {
  return getToken(GpuSqlParser::ATAN, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::LOG10() {
  return getToken(GpuSqlParser::LOG10, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::LOG() {
  return getToken(GpuSqlParser::LOG, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::EXP() {
  return getToken(GpuSqlParser::EXP, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::SQRT() {
  return getToken(GpuSqlParser::SQRT, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::SQUARE() {
  return getToken(GpuSqlParser::SQUARE, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::SIGN() {
  return getToken(GpuSqlParser::SIGN, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::ROUND() {
  return getToken(GpuSqlParser::ROUND, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::FLOOR() {
  return getToken(GpuSqlParser::FLOOR, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::CEIL() {
  return getToken(GpuSqlParser::CEIL, 0);
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

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::LTRIM() {
  return getToken(GpuSqlParser::LTRIM, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::RTRIM() {
  return getToken(GpuSqlParser::RTRIM, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::LOWER() {
  return getToken(GpuSqlParser::LOWER, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::UPPER() {
  return getToken(GpuSqlParser::UPPER, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::REVERSE() {
  return getToken(GpuSqlParser::REVERSE, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::LEN() {
  return getToken(GpuSqlParser::LEN, 0);
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
//----------------- PiLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::PiLiteralContext::PI() {
  return getToken(GpuSqlParser::PI, 0);
}

GpuSqlParser::PiLiteralContext::PiLiteralContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::PiLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPiLiteral(this);
}
void GpuSqlParser::PiLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPiLiteral(this);
}
//----------------- StringLiteralContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::StringLiteralContext::STRING() {
  return getToken(GpuSqlParser::STRING, 0);
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

GpuSqlParser::ExpressionContext* GpuSqlParser::expression() {
   return expression(0);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  GpuSqlParser::ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, parentState);
  GpuSqlParser::ExpressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 96;
  enterRecursionRule(_localctx, 96, GpuSqlParser::RuleExpression, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(647);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(403);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOT);
      setState(404);
      expression(66);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(405);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUS);
      setState(406);
      expression(65);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(407);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ABS);
      setState(408);
      match(GpuSqlParser::LPAREN);
      setState(409);
      expression(0);
      setState(410);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(412);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIN);
      setState(413);
      match(GpuSqlParser::LPAREN);
      setState(414);
      expression(0);
      setState(415);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(417);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COS);
      setState(418);
      match(GpuSqlParser::LPAREN);
      setState(419);
      expression(0);
      setState(420);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(422);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::TAN);
      setState(423);
      match(GpuSqlParser::LPAREN);
      setState(424);
      expression(0);
      setState(425);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(427);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COT);
      setState(428);
      match(GpuSqlParser::LPAREN);
      setState(429);
      expression(0);
      setState(430);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(432);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ASIN);
      setState(433);
      match(GpuSqlParser::LPAREN);
      setState(434);
      expression(0);
      setState(435);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(437);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ACOS);
      setState(438);
      match(GpuSqlParser::LPAREN);
      setState(439);
      expression(0);
      setState(440);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(442);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN);
      setState(443);
      match(GpuSqlParser::LPAREN);
      setState(444);
      expression(0);
      setState(445);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 11: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(447);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG10);
      setState(448);
      match(GpuSqlParser::LPAREN);
      setState(449);
      expression(0);
      setState(450);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 12: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(452);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(453);
      match(GpuSqlParser::LPAREN);
      setState(454);
      expression(0);
      setState(455);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 13: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(457);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::EXP);
      setState(458);
      match(GpuSqlParser::LPAREN);
      setState(459);
      expression(0);
      setState(460);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 14: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(462);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQRT);
      setState(463);
      match(GpuSqlParser::LPAREN);
      setState(464);
      expression(0);
      setState(465);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 15: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(467);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQUARE);
      setState(468);
      match(GpuSqlParser::LPAREN);
      setState(469);
      expression(0);
      setState(470);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 16: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(472);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIGN);
      setState(473);
      match(GpuSqlParser::LPAREN);
      setState(474);
      expression(0);
      setState(475);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 17: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(477);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROUND);
      setState(478);
      match(GpuSqlParser::LPAREN);
      setState(479);
      expression(0);
      setState(480);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 18: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(482);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::FLOOR);
      setState(483);
      match(GpuSqlParser::LPAREN);
      setState(484);
      expression(0);
      setState(485);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 19: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(487);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::CEIL);
      setState(488);
      match(GpuSqlParser::LPAREN);
      setState(489);
      expression(0);
      setState(490);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 20: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(492);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::YEAR);
      setState(493);
      match(GpuSqlParser::LPAREN);
      setState(494);
      expression(0);
      setState(495);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 21: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(497);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MONTH);
      setState(498);
      match(GpuSqlParser::LPAREN);
      setState(499);
      expression(0);
      setState(500);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 22: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(502);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DAY);
      setState(503);
      match(GpuSqlParser::LPAREN);
      setState(504);
      expression(0);
      setState(505);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 23: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(507);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::HOUR);
      setState(508);
      match(GpuSqlParser::LPAREN);
      setState(509);
      expression(0);
      setState(510);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 24: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(512);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUTE);
      setState(513);
      match(GpuSqlParser::LPAREN);
      setState(514);
      expression(0);
      setState(515);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 25: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(517);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SECOND);
      setState(518);
      match(GpuSqlParser::LPAREN);
      setState(519);
      expression(0);
      setState(520);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 26: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(522);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LTRIM);
      setState(523);
      match(GpuSqlParser::LPAREN);
      setState(524);
      expression(0);
      setState(525);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 27: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(527);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::RTRIM);
      setState(528);
      match(GpuSqlParser::LPAREN);
      setState(529);
      expression(0);
      setState(530);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 28: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(532);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOWER);
      setState(533);
      match(GpuSqlParser::LPAREN);
      setState(534);
      expression(0);
      setState(535);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 29: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(537);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::UPPER);
      setState(538);
      match(GpuSqlParser::LPAREN);
      setState(539);
      expression(0);
      setState(540);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 30: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(542);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::REVERSE);
      setState(543);
      match(GpuSqlParser::LPAREN);
      setState(544);
      expression(0);
      setState(545);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 31: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(547);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LEN);
      setState(548);
      match(GpuSqlParser::LPAREN);
      setState(549);
      expression(0);
      setState(550);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 32: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(552);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN2);
      setState(553);
      match(GpuSqlParser::LPAREN);
      setState(554);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(555);
      match(GpuSqlParser::COMMA);
      setState(556);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(557);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 33: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(559);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(560);
      match(GpuSqlParser::LPAREN);
      setState(561);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(562);
      match(GpuSqlParser::COMMA);
      setState(563);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(564);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 34: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(566);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POW);
      setState(567);
      match(GpuSqlParser::LPAREN);
      setState(568);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(569);
      match(GpuSqlParser::COMMA);
      setState(570);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(571);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 35: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(573);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROOT);
      setState(574);
      match(GpuSqlParser::LPAREN);
      setState(575);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(576);
      match(GpuSqlParser::COMMA);
      setState(577);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(578);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 36: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(580);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POINT);
      setState(581);
      match(GpuSqlParser::LPAREN);
      setState(582);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(583);
      match(GpuSqlParser::COMMA);
      setState(584);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(585);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 37: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(587);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_CONTAINS);
      setState(588);
      match(GpuSqlParser::LPAREN);
      setState(589);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(590);
      match(GpuSqlParser::COMMA);
      setState(591);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(592);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 38: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(594);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_INTERSECT);
      setState(595);
      match(GpuSqlParser::LPAREN);
      setState(596);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(597);
      match(GpuSqlParser::COMMA);
      setState(598);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(599);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 39: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(601);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_UNION);
      setState(602);
      match(GpuSqlParser::LPAREN);
      setState(603);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(604);
      match(GpuSqlParser::COMMA);
      setState(605);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(606);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 40: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(608);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::CONCAT);
      setState(609);
      match(GpuSqlParser::LPAREN);
      setState(610);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(611);
      match(GpuSqlParser::COMMA);
      setState(612);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(613);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 41: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(615);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LEFT);
      setState(616);
      match(GpuSqlParser::LPAREN);
      setState(617);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(618);
      match(GpuSqlParser::COMMA);
      setState(619);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(620);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 42: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(622);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::RIGHT);
      setState(623);
      match(GpuSqlParser::LPAREN);
      setState(624);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(625);
      match(GpuSqlParser::COMMA);
      setState(626);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(627);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 43: {
      _localctx = _tracker.createInstance<ParenExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(629);
      match(GpuSqlParser::LPAREN);
      setState(630);
      expression(0);
      setState(631);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 44: {
      _localctx = _tracker.createInstance<VarReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(633);
      columnId();
      break;
    }

    case 45: {
      _localctx = _tracker.createInstance<GeoReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(634);
      geometry();
      break;
    }

    case 46: {
      _localctx = _tracker.createInstance<DateTimeLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(635);
      match(GpuSqlParser::DATETIMELIT);
      break;
    }

    case 47: {
      _localctx = _tracker.createInstance<DecimalLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(636);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 48: {
      _localctx = _tracker.createInstance<PiLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(637);
      match(GpuSqlParser::PI);
      break;
    }

    case 49: {
      _localctx = _tracker.createInstance<NowLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(638);
      match(GpuSqlParser::NOW);
      break;
    }

    case 50: {
      _localctx = _tracker.createInstance<IntLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(639);
      match(GpuSqlParser::INTLIT);
      break;
    }

    case 51: {
      _localctx = _tracker.createInstance<StringLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(640);
      match(GpuSqlParser::STRING);
      break;
    }

    case 52: {
      _localctx = _tracker.createInstance<BooleanLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(641);
      match(GpuSqlParser::BOOLEANLIT);
      break;
    }

    case 53: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(642);
      match(GpuSqlParser::AGG);
      setState(643);
      match(GpuSqlParser::LPAREN);
      setState(644);
      expression(0);
      setState(645);
      match(GpuSqlParser::RPAREN);
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(693);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 32, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(691);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(649);

          if (!(precpred(_ctx, 35))) throw FailedPredicateException(this, "precpred(_ctx, 35)");
          setState(650);
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
          setState(651);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(36);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(652);

          if (!(precpred(_ctx, 34))) throw FailedPredicateException(this, "precpred(_ctx, 34)");
          setState(653);
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
          setState(654);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(35);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(655);

          if (!(precpred(_ctx, 33))) throw FailedPredicateException(this, "precpred(_ctx, 33)");
          setState(656);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MODULO);
          setState(657);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(34);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(658);

          if (!(precpred(_ctx, 28))) throw FailedPredicateException(this, "precpred(_ctx, 28)");
          setState(659);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::XOR);
          setState(660);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(29);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(661);

          if (!(precpred(_ctx, 27))) throw FailedPredicateException(this, "precpred(_ctx, 27)");
          setState(662);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == GpuSqlParser::BIT_OR

          || _la == GpuSqlParser::BIT_AND)) {
            dynamic_cast<BinaryOperationContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(663);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(28);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(664);

          if (!(precpred(_ctx, 26))) throw FailedPredicateException(this, "precpred(_ctx, 26)");
          setState(665);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == GpuSqlParser::L_SHIFT

          || _la == GpuSqlParser::R_SHIFT)) {
            dynamic_cast<BinaryOperationContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(666);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(27);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(667);

          if (!(precpred(_ctx, 25))) throw FailedPredicateException(this, "precpred(_ctx, 25)");
          setState(668);
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
          setState(669);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(26);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(670);

          if (!(precpred(_ctx, 24))) throw FailedPredicateException(this, "precpred(_ctx, 24)");
          setState(671);
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
          setState(672);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(25);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(673);

          if (!(precpred(_ctx, 23))) throw FailedPredicateException(this, "precpred(_ctx, 23)");
          setState(674);
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
          setState(675);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(24);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(676);

          if (!(precpred(_ctx, 22))) throw FailedPredicateException(this, "precpred(_ctx, 22)");
          setState(677);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOTEQUALS_GT_LT);
          setState(678);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(23);
          break;
        }

        case 11: {
          auto newContext = _tracker.createInstance<TernaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(679);

          if (!(precpred(_ctx, 14))) throw FailedPredicateException(this, "precpred(_ctx, 14)");
          setState(680);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op = match(GpuSqlParser::BETWEEN);
          setState(681);
          expression(0);
          setState(682);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op2 = match(GpuSqlParser::AND);
          setState(683);
          expression(15);
          break;
        }

        case 12: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(685);

          if (!(precpred(_ctx, 13))) throw FailedPredicateException(this, "precpred(_ctx, 13)");
          setState(686);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::AND);
          setState(687);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(14);
          break;
        }

        case 13: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(688);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(689);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::OR);
          setState(690);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(13);
          break;
        }

        } 
      }
      setState(695);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 32, _ctx);
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
  enterRule(_localctx, 98, GpuSqlParser::RuleGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(702);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::POINT: {
        setState(696);
        pointGeometry();
        break;
      }

      case GpuSqlParser::POLYGON: {
        setState(697);
        polygonGeometry();
        break;
      }

      case GpuSqlParser::LINESTRING: {
        setState(698);
        lineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOINT: {
        setState(699);
        multiPointGeometry();
        break;
      }

      case GpuSqlParser::MULTILINESTRING: {
        setState(700);
        multiLineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOLYGON: {
        setState(701);
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
  enterRule(_localctx, 100, GpuSqlParser::RulePointGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(704);
    match(GpuSqlParser::POINT);
    setState(705);
    match(GpuSqlParser::LPAREN);
    setState(706);
    point();
    setState(707);
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
  enterRule(_localctx, 102, GpuSqlParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(709);
    match(GpuSqlParser::LINESTRING);
    setState(710);
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
  enterRule(_localctx, 104, GpuSqlParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(712);
    match(GpuSqlParser::POLYGON);
    setState(713);
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
  enterRule(_localctx, 106, GpuSqlParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(715);
    match(GpuSqlParser::MULTIPOINT);
    setState(716);
    match(GpuSqlParser::LPAREN);
    setState(717);
    pointOrClosedPoint();
    setState(722);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(718);
      match(GpuSqlParser::COMMA);
      setState(719);
      pointOrClosedPoint();
      setState(724);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(725);
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
  enterRule(_localctx, 108, GpuSqlParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(727);
    match(GpuSqlParser::MULTILINESTRING);
    setState(728);
    match(GpuSqlParser::LPAREN);
    setState(729);
    lineString();
    setState(734);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(730);
      match(GpuSqlParser::COMMA);
      setState(731);
      lineString();
      setState(736);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(737);
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
  enterRule(_localctx, 110, GpuSqlParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(739);
    match(GpuSqlParser::MULTIPOLYGON);
    setState(740);
    match(GpuSqlParser::LPAREN);
    setState(741);
    polygon();
    setState(746);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(742);
      match(GpuSqlParser::COMMA);
      setState(743);
      polygon();
      setState(748);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(749);
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
  enterRule(_localctx, 112, GpuSqlParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(756);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT: {
        enterOuterAlt(_localctx, 1);
        setState(751);
        point();
        break;
      }

      case GpuSqlParser::LPAREN: {
        enterOuterAlt(_localctx, 2);
        setState(752);
        match(GpuSqlParser::LPAREN);
        setState(753);
        point();
        setState(754);
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
  enterRule(_localctx, 114, GpuSqlParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(758);
    match(GpuSqlParser::LPAREN);
    setState(759);
    lineString();
    setState(764);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(760);
      match(GpuSqlParser::COMMA);
      setState(761);
      lineString();
      setState(766);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(767);
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
  enterRule(_localctx, 116, GpuSqlParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(769);
    match(GpuSqlParser::LPAREN);
    setState(770);
    point();
    setState(775);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(771);
      match(GpuSqlParser::COMMA);
      setState(772);
      point();
      setState(777);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(778);
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
  enterRule(_localctx, 118, GpuSqlParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(780);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FLOATLIT

    || _la == GpuSqlParser::INTLIT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(781);
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
    case 48: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool GpuSqlParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 35);
    case 1: return precpred(_ctx, 34);
    case 2: return precpred(_ctx, 33);
    case 3: return precpred(_ctx, 28);
    case 4: return precpred(_ctx, 27);
    case 5: return precpred(_ctx, 26);
    case 6: return precpred(_ctx, 25);
    case 7: return precpred(_ctx, 24);
    case 8: return precpred(_ctx, 23);
    case 9: return precpred(_ctx, 22);
    case 10: return precpred(_ctx, 14);
    case 11: return precpred(_ctx, 13);
    case 12: return precpred(_ctx, 12);

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
  "showColumns", "sqlSelect", "sqlCreateDb", "sqlDropDb", "sqlCreateTable", 
  "sqlDropTable", "sqlAlterTable", "sqlCreateIndex", "sqlInsertInto", "newTableEntries", 
  "newTableEntry", "alterTableEntries", "alterTableEntry", "addColumn", 
  "dropColumn", "alterColumn", "newTableColumn", "newTableIndex", "selectColumns", 
  "selectColumn", "whereClause", "orderByColumns", "orderByColumn", "insertIntoValues", 
  "insertIntoColumns", "indexColumns", "groupByColumns", "groupByColumn", 
  "columnId", "fromTables", "joinClauses", "joinClause", "joinTable", "fromTable", 
  "table", "column", "database", "alias", "indexName", "limit", "offset", 
  "blockSize", "columnValue", "expression", "geometry", "pointGeometry", 
  "lineStringGeometry", "polygonGeometry", "multiPointGeometry", "multiLineStringGeometry", 
  "multiPolygonGeometry", "pointOrClosedPoint", "polygon", "lineString", 
  "point"
};

std::vector<std::string> GpuSqlParser::_literalNames = {
  "", "", "'\n'", "'\r'", "'\r\n'", "", "';'", "'''", "'\"'", "'_'", "':'", 
  "','", "'.'", "", "", "", "", "'POINT'", "'MULTIPOINT'", "'LINESTRING'", 
  "'MULTILINESTRING'", "'POLYGON'", "'MULTIPOLYGON'", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "'+'", "'-'", "'*'", "'/'", "'%'", "'^'", 
  "'='", "'!='", "'<>'", "'('", "')'", "'>'", "'<'", "'>='", "'<='", "'!'", 
  "", "", "'|'", "'&'", "'<<'", "'>>'"
};

std::vector<std::string> GpuSqlParser::_symbolicNames = {
  "", "DATETIMELIT", "LF", "CR", "CRLF", "WS", "SEMICOL", "SQOUTE", "DQOUTE", 
  "UNDERSCORE", "COLON", "COMMA", "DOT", "STRING", "DELIMID", "DATELIT", 
  "TIMELIT", "POINT", "MULTIPOINT", "LINESTRING", "MULTILINESTRING", "POLYGON", 
  "MULTIPOLYGON", "DATATYPE", "INTTYPE", "LONGTYPE", "FLOATTYPE", "DOUBLETYPE", 
  "STRINGTYPE", "BOOLEANTYPE", "POINTTYPE", "POLYTYPE", "INSERTINTO", "CREATEDB", 
  "DROPDB", "CREATETABLE", "DROPTABLE", "ALTERTABLE", "ADD", "DROPCOLUMN", 
  "ALTERCOLUMN", "CREATEINDEX", "INDEX", "PRIMARYKEY", "VALUES", "SELECT", 
  "FROM", "JOIN", "WHERE", "GROUPBY", "AS", "IN", "BETWEEN", "ON", "ORDERBY", 
  "DIR", "LIMIT", "OFFSET", "SHOWDB", "SHOWTB", "SHOWCL", "AGG", "AVG", 
  "SUM", "MIN", "MAX", "COUNT", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", 
  "SECOND", "NOW", "PI", "ABS", "SIN", "COS", "TAN", "COT", "ASIN", "ACOS", 
  "ATAN", "ATAN2", "LOG10", "LOG", "EXP", "POW", "SQRT", "SQUARE", "SIGN", 
  "ROOT", "ROUND", "CEIL", "FLOOR", "LTRIM", "RTRIM", "LOWER", "UPPER", 
  "REVERSE", "LEN", "LEFT", "RIGHT", "CONCAT", "GEO_CONTAINS", "GEO_INTERSECT", 
  "GEO_UNION", "PLUS", "MINUS", "ASTERISK", "DIVISION", "MODULO", "XOR", 
  "EQUALS", "NOTEQUALS", "NOTEQUALS_GT_LT", "LPAREN", "RPAREN", "GREATER", 
  "LESS", "GREATEREQ", "LESSEQ", "NOT", "OR", "AND", "BIT_OR", "BIT_AND", 
  "L_SHIFT", "R_SHIFT", "BOOLEANLIT", "TRUE", "FALSE", "FLOATLIT", "INTLIT", 
  "ID"
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
    0x3, 0x88, 0x312, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x2c, 0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x4, 0x2e, 0x9, 0x2e, 0x4, 0x2f, 
    0x9, 0x2f, 0x4, 0x30, 0x9, 0x30, 0x4, 0x31, 0x9, 0x31, 0x4, 0x32, 0x9, 
    0x32, 0x4, 0x33, 0x9, 0x33, 0x4, 0x34, 0x9, 0x34, 0x4, 0x35, 0x9, 0x35, 
    0x4, 0x36, 0x9, 0x36, 0x4, 0x37, 0x9, 0x37, 0x4, 0x38, 0x9, 0x38, 0x4, 
    0x39, 0x9, 0x39, 0x4, 0x3a, 0x9, 0x3a, 0x4, 0x3b, 0x9, 0x3b, 0x4, 0x3c, 
    0x9, 0x3c, 0x4, 0x3d, 0x9, 0x3d, 0x3, 0x2, 0x7, 0x2, 0x7c, 0xa, 0x2, 
    0xc, 0x2, 0xe, 0x2, 0x7f, 0xb, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x5, 0x3, 0x8c, 0xa, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 
    0x91, 0xa, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x5, 0x6, 0x99, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0xa2, 0xa, 0x7, 0x3, 0x7, 0x3, 
    0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xab, 
    0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xaf, 0xa, 0x8, 0x3, 0x8, 0x3, 
    0x8, 0x5, 0x8, 0xb3, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb7, 0xa, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xbb, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x5, 0x8, 0xbf, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x5, 0x9, 0xc6, 0xa, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x7, 
    0x10, 0xf5, 0xa, 0x10, 0xc, 0x10, 0xe, 0x10, 0xf8, 0xb, 0x10, 0x3, 0x11, 
    0x3, 0x11, 0x5, 0x11, 0xfc, 0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x7, 0x12, 0x101, 0xa, 0x12, 0xc, 0x12, 0xe, 0x12, 0x104, 0xb, 0x12, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x5, 0x13, 0x109, 0xa, 0x13, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 
    0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 
    0x19, 0x3, 0x19, 0x3, 0x19, 0x7, 0x19, 0x122, 0xa, 0x19, 0xc, 0x19, 
    0xe, 0x19, 0x125, 0xb, 0x19, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 
    0x12a, 0xa, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
    0x7, 0x1c, 0x131, 0xa, 0x1c, 0xc, 0x1c, 0xe, 0x1c, 0x134, 0xb, 0x1c, 
    0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x138, 0xa, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x7, 0x1e, 0x13d, 0xa, 0x1e, 0xc, 0x1e, 0xe, 0x1e, 0x140, 
    0xb, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x7, 0x1f, 0x145, 0xa, 0x1f, 
    0xc, 0x1f, 0xe, 0x1f, 0x148, 0xb, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
    0x7, 0x20, 0x14d, 0xa, 0x20, 0xc, 0x20, 0xe, 0x20, 0x150, 0xb, 0x20, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x7, 0x21, 0x155, 0xa, 0x21, 0xc, 0x21, 
    0xe, 0x21, 0x158, 0xb, 0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x5, 0x23, 0x161, 0xa, 0x23, 0x3, 0x24, 
    0x3, 0x24, 0x3, 0x24, 0x7, 0x24, 0x166, 0xa, 0x24, 0xc, 0x24, 0xe, 0x24, 
    0x169, 0xb, 0x24, 0x3, 0x25, 0x6, 0x25, 0x16c, 0xa, 0x25, 0xd, 0x25, 
    0xe, 0x25, 0x16d, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 
    0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x5, 0x27, 0x178, 0xa, 0x27, 0x3, 0x28, 
    0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 0x17d, 0xa, 0x28, 0x3, 0x29, 0x3, 0x29, 
    0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 
    0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x30, 
    0x3, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x5, 0x31, 0x193, 
    0xa, 0x31, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x5, 0x32, 
    0x28a, 0xa, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x7, 0x32, 0x2b6, 
    0xa, 0x32, 0xc, 0x32, 0xe, 0x32, 0x2b9, 0xb, 0x32, 0x3, 0x33, 0x3, 0x33, 
    0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x5, 0x33, 0x2c1, 0xa, 0x33, 
    0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x35, 0x3, 
    0x35, 0x3, 0x35, 0x3, 0x36, 0x3, 0x36, 0x3, 0x36, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x7, 0x37, 0x2d3, 0xa, 0x37, 0xc, 0x37, 
    0xe, 0x37, 0x2d6, 0xb, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x38, 0x3, 0x38, 
    0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x7, 0x38, 0x2df, 0xa, 0x38, 0xc, 0x38, 
    0xe, 0x38, 0x2e2, 0xb, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x39, 0x3, 0x39, 
    0x3, 0x39, 0x3, 0x39, 0x3, 0x39, 0x7, 0x39, 0x2eb, 0xa, 0x39, 0xc, 0x39, 
    0xe, 0x39, 0x2ee, 0xb, 0x39, 0x3, 0x39, 0x3, 0x39, 0x3, 0x3a, 0x3, 0x3a, 
    0x3, 0x3a, 0x3, 0x3a, 0x3, 0x3a, 0x5, 0x3a, 0x2f7, 0xa, 0x3a, 0x3, 0x3b, 
    0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x7, 0x3b, 0x2fd, 0xa, 0x3b, 0xc, 0x3b, 
    0xe, 0x3b, 0x300, 0xb, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3c, 0x3, 0x3c, 
    0x3, 0x3c, 0x3, 0x3c, 0x7, 0x3c, 0x308, 0xa, 0x3c, 0xc, 0x3c, 0xe, 0x3c, 
    0x30b, 0xb, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 
    0x3, 0x3d, 0x2, 0x3, 0x62, 0x3e, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 
    0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 
    0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 
    0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 
    0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 
    0x70, 0x72, 0x74, 0x76, 0x78, 0x2, 0xc, 0x4, 0x2, 0x30, 0x30, 0x35, 
    0x35, 0x4, 0x2, 0x10, 0x10, 0x88, 0x88, 0x3, 0x2, 0x6f, 0x70, 0x3, 0x2, 
    0x6d, 0x6e, 0x3, 0x2, 0x7f, 0x80, 0x3, 0x2, 0x81, 0x82, 0x3, 0x2, 0x78, 
    0x79, 0x3, 0x2, 0x7a, 0x7b, 0x3, 0x2, 0x73, 0x74, 0x3, 0x2, 0x86, 0x87, 
    0x2, 0x34a, 0x2, 0x7d, 0x3, 0x2, 0x2, 0x2, 0x4, 0x8b, 0x3, 0x2, 0x2, 
    0x2, 0x6, 0x90, 0x3, 0x2, 0x2, 0x2, 0x8, 0x92, 0x3, 0x2, 0x2, 0x2, 0xa, 
    0x95, 0x3, 0x2, 0x2, 0x2, 0xc, 0x9c, 0x3, 0x2, 0x2, 0x2, 0xe, 0xa5, 
    0x3, 0x2, 0x2, 0x2, 0x10, 0xc2, 0x3, 0x2, 0x2, 0x2, 0x12, 0xc9, 0x3, 
    0x2, 0x2, 0x2, 0x14, 0xcd, 0x3, 0x2, 0x2, 0x2, 0x16, 0xd4, 0x3, 0x2, 
    0x2, 0x2, 0x18, 0xd8, 0x3, 0x2, 0x2, 0x2, 0x1a, 0xdd, 0x3, 0x2, 0x2, 
    0x2, 0x1c, 0xe6, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xf1, 0x3, 0x2, 0x2, 0x2, 
    0x20, 0xfb, 0x3, 0x2, 0x2, 0x2, 0x22, 0xfd, 0x3, 0x2, 0x2, 0x2, 0x24, 
    0x108, 0x3, 0x2, 0x2, 0x2, 0x26, 0x10a, 0x3, 0x2, 0x2, 0x2, 0x28, 0x10e, 
    0x3, 0x2, 0x2, 0x2, 0x2a, 0x111, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x115, 0x3, 
    0x2, 0x2, 0x2, 0x2e, 0x118, 0x3, 0x2, 0x2, 0x2, 0x30, 0x11e, 0x3, 0x2, 
    0x2, 0x2, 0x32, 0x126, 0x3, 0x2, 0x2, 0x2, 0x34, 0x12b, 0x3, 0x2, 0x2, 
    0x2, 0x36, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x38, 0x135, 0x3, 0x2, 0x2, 0x2, 
    0x3a, 0x139, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x141, 0x3, 0x2, 0x2, 0x2, 0x3e, 
    0x149, 0x3, 0x2, 0x2, 0x2, 0x40, 0x151, 0x3, 0x2, 0x2, 0x2, 0x42, 0x159, 
    0x3, 0x2, 0x2, 0x2, 0x44, 0x160, 0x3, 0x2, 0x2, 0x2, 0x46, 0x162, 0x3, 
    0x2, 0x2, 0x2, 0x48, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x16f, 0x3, 0x2, 
    0x2, 0x2, 0x4c, 0x174, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x179, 0x3, 0x2, 0x2, 
    0x2, 0x50, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x52, 0x180, 0x3, 0x2, 0x2, 0x2, 
    0x54, 0x182, 0x3, 0x2, 0x2, 0x2, 0x56, 0x184, 0x3, 0x2, 0x2, 0x2, 0x58, 
    0x186, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x188, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x18a, 
    0x3, 0x2, 0x2, 0x2, 0x5e, 0x18c, 0x3, 0x2, 0x2, 0x2, 0x60, 0x192, 0x3, 
    0x2, 0x2, 0x2, 0x62, 0x289, 0x3, 0x2, 0x2, 0x2, 0x64, 0x2c0, 0x3, 0x2, 
    0x2, 0x2, 0x66, 0x2c2, 0x3, 0x2, 0x2, 0x2, 0x68, 0x2c7, 0x3, 0x2, 0x2, 
    0x2, 0x6a, 0x2ca, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x2cd, 0x3, 0x2, 0x2, 0x2, 
    0x6e, 0x2d9, 0x3, 0x2, 0x2, 0x2, 0x70, 0x2e5, 0x3, 0x2, 0x2, 0x2, 0x72, 
    0x2f6, 0x3, 0x2, 0x2, 0x2, 0x74, 0x2f8, 0x3, 0x2, 0x2, 0x2, 0x76, 0x303, 
    0x3, 0x2, 0x2, 0x2, 0x78, 0x30e, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x7c, 0x5, 
    0x4, 0x3, 0x2, 0x7b, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x7f, 0x3, 0x2, 
    0x2, 0x2, 0x7d, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x7e, 0x3, 0x2, 0x2, 
    0x2, 0x7e, 0x80, 0x3, 0x2, 0x2, 0x2, 0x7f, 0x7d, 0x3, 0x2, 0x2, 0x2, 
    0x80, 0x81, 0x7, 0x2, 0x2, 0x3, 0x81, 0x3, 0x3, 0x2, 0x2, 0x2, 0x82, 
    0x8c, 0x5, 0xe, 0x8, 0x2, 0x83, 0x8c, 0x5, 0x10, 0x9, 0x2, 0x84, 0x8c, 
    0x5, 0x12, 0xa, 0x2, 0x85, 0x8c, 0x5, 0x14, 0xb, 0x2, 0x86, 0x8c, 0x5, 
    0x16, 0xc, 0x2, 0x87, 0x8c, 0x5, 0x18, 0xd, 0x2, 0x88, 0x8c, 0x5, 0x1a, 
    0xe, 0x2, 0x89, 0x8c, 0x5, 0x1c, 0xf, 0x2, 0x8a, 0x8c, 0x5, 0x6, 0x4, 
    0x2, 0x8b, 0x82, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x83, 0x3, 0x2, 0x2, 0x2, 
    0x8b, 0x84, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x85, 0x3, 0x2, 0x2, 0x2, 0x8b, 
    0x86, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x87, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x88, 
    0x3, 0x2, 0x2, 0x2, 0x8b, 0x89, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x8a, 0x3, 
    0x2, 0x2, 0x2, 0x8c, 0x5, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x91, 0x5, 0x8, 
    0x5, 0x2, 0x8e, 0x91, 0x5, 0xa, 0x6, 0x2, 0x8f, 0x91, 0x5, 0xc, 0x7, 
    0x2, 0x90, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x90, 0x8e, 0x3, 0x2, 0x2, 0x2, 
    0x90, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x91, 0x7, 0x3, 0x2, 0x2, 0x2, 0x92, 
    0x93, 0x7, 0x3c, 0x2, 0x2, 0x93, 0x94, 0x7, 0x8, 0x2, 0x2, 0x94, 0x9, 
    0x3, 0x2, 0x2, 0x2, 0x95, 0x98, 0x7, 0x3d, 0x2, 0x2, 0x96, 0x97, 0x9, 
    0x2, 0x2, 0x2, 0x97, 0x99, 0x5, 0x54, 0x2b, 0x2, 0x98, 0x96, 0x3, 0x2, 
    0x2, 0x2, 0x98, 0x99, 0x3, 0x2, 0x2, 0x2, 0x99, 0x9a, 0x3, 0x2, 0x2, 
    0x2, 0x9a, 0x9b, 0x7, 0x8, 0x2, 0x2, 0x9b, 0xb, 0x3, 0x2, 0x2, 0x2, 
    0x9c, 0x9d, 0x7, 0x3e, 0x2, 0x2, 0x9d, 0x9e, 0x9, 0x2, 0x2, 0x2, 0x9e, 
    0xa1, 0x5, 0x50, 0x29, 0x2, 0x9f, 0xa0, 0x9, 0x2, 0x2, 0x2, 0xa0, 0xa2, 
    0x5, 0x54, 0x2b, 0x2, 0xa1, 0x9f, 0x3, 0x2, 0x2, 0x2, 0xa1, 0xa2, 0x3, 
    0x2, 0x2, 0x2, 0xa2, 0xa3, 0x3, 0x2, 0x2, 0x2, 0xa3, 0xa4, 0x7, 0x8, 
    0x2, 0x2, 0xa4, 0xd, 0x3, 0x2, 0x2, 0x2, 0xa5, 0xa6, 0x7, 0x2f, 0x2, 
    0x2, 0xa6, 0xa7, 0x5, 0x30, 0x19, 0x2, 0xa7, 0xa8, 0x7, 0x30, 0x2, 0x2, 
    0xa8, 0xaa, 0x5, 0x46, 0x24, 0x2, 0xa9, 0xab, 0x5, 0x48, 0x25, 0x2, 
    0xaa, 0xa9, 0x3, 0x2, 0x2, 0x2, 0xaa, 0xab, 0x3, 0x2, 0x2, 0x2, 0xab, 
    0xae, 0x3, 0x2, 0x2, 0x2, 0xac, 0xad, 0x7, 0x32, 0x2, 0x2, 0xad, 0xaf, 
    0x5, 0x34, 0x1b, 0x2, 0xae, 0xac, 0x3, 0x2, 0x2, 0x2, 0xae, 0xaf, 0x3, 
    0x2, 0x2, 0x2, 0xaf, 0xb2, 0x3, 0x2, 0x2, 0x2, 0xb0, 0xb1, 0x7, 0x33, 
    0x2, 0x2, 0xb1, 0xb3, 0x5, 0x40, 0x21, 0x2, 0xb2, 0xb0, 0x3, 0x2, 0x2, 
    0x2, 0xb2, 0xb3, 0x3, 0x2, 0x2, 0x2, 0xb3, 0xb6, 0x3, 0x2, 0x2, 0x2, 
    0xb4, 0xb5, 0x7, 0x38, 0x2, 0x2, 0xb5, 0xb7, 0x5, 0x36, 0x1c, 0x2, 0xb6, 
    0xb4, 0x3, 0x2, 0x2, 0x2, 0xb6, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xb7, 0xba, 
    0x3, 0x2, 0x2, 0x2, 0xb8, 0xb9, 0x7, 0x3a, 0x2, 0x2, 0xb9, 0xbb, 0x5, 
    0x5a, 0x2e, 0x2, 0xba, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xba, 0xbb, 0x3, 0x2, 
    0x2, 0x2, 0xbb, 0xbe, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xbd, 0x7, 0x3b, 0x2, 
    0x2, 0xbd, 0xbf, 0x5, 0x5c, 0x2f, 0x2, 0xbe, 0xbc, 0x3, 0x2, 0x2, 0x2, 
    0xbe, 0xbf, 0x3, 0x2, 0x2, 0x2, 0xbf, 0xc0, 0x3, 0x2, 0x2, 0x2, 0xc0, 
    0xc1, 0x7, 0x8, 0x2, 0x2, 0xc1, 0xf, 0x3, 0x2, 0x2, 0x2, 0xc2, 0xc3, 
    0x7, 0x23, 0x2, 0x2, 0xc3, 0xc5, 0x5, 0x54, 0x2b, 0x2, 0xc4, 0xc6, 0x5, 
    0x5e, 0x30, 0x2, 0xc5, 0xc4, 0x3, 0x2, 0x2, 0x2, 0xc5, 0xc6, 0x3, 0x2, 
    0x2, 0x2, 0xc6, 0xc7, 0x3, 0x2, 0x2, 0x2, 0xc7, 0xc8, 0x7, 0x8, 0x2, 
    0x2, 0xc8, 0x11, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xca, 0x7, 0x24, 0x2, 0x2, 
    0xca, 0xcb, 0x5, 0x54, 0x2b, 0x2, 0xcb, 0xcc, 0x7, 0x8, 0x2, 0x2, 0xcc, 
    0x13, 0x3, 0x2, 0x2, 0x2, 0xcd, 0xce, 0x7, 0x25, 0x2, 0x2, 0xce, 0xcf, 
    0x5, 0x50, 0x29, 0x2, 0xcf, 0xd0, 0x7, 0x76, 0x2, 0x2, 0xd0, 0xd1, 0x5, 
    0x1e, 0x10, 0x2, 0xd1, 0xd2, 0x7, 0x77, 0x2, 0x2, 0xd2, 0xd3, 0x7, 0x8, 
    0x2, 0x2, 0xd3, 0x15, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd5, 0x7, 0x26, 0x2, 
    0x2, 0xd5, 0xd6, 0x5, 0x50, 0x29, 0x2, 0xd6, 0xd7, 0x7, 0x8, 0x2, 0x2, 
    0xd7, 0x17, 0x3, 0x2, 0x2, 0x2, 0xd8, 0xd9, 0x7, 0x27, 0x2, 0x2, 0xd9, 
    0xda, 0x5, 0x50, 0x29, 0x2, 0xda, 0xdb, 0x5, 0x22, 0x12, 0x2, 0xdb, 
    0xdc, 0x7, 0x8, 0x2, 0x2, 0xdc, 0x19, 0x3, 0x2, 0x2, 0x2, 0xdd, 0xde, 
    0x7, 0x2b, 0x2, 0x2, 0xde, 0xdf, 0x5, 0x58, 0x2d, 0x2, 0xdf, 0xe0, 0x7, 
    0x37, 0x2, 0x2, 0xe0, 0xe1, 0x5, 0x50, 0x29, 0x2, 0xe1, 0xe2, 0x7, 0x76, 
    0x2, 0x2, 0xe2, 0xe3, 0x5, 0x3e, 0x20, 0x2, 0xe3, 0xe4, 0x7, 0x77, 0x2, 
    0x2, 0xe4, 0xe5, 0x7, 0x8, 0x2, 0x2, 0xe5, 0x1b, 0x3, 0x2, 0x2, 0x2, 
    0xe6, 0xe7, 0x7, 0x22, 0x2, 0x2, 0xe7, 0xe8, 0x5, 0x50, 0x29, 0x2, 0xe8, 
    0xe9, 0x7, 0x76, 0x2, 0x2, 0xe9, 0xea, 0x5, 0x3c, 0x1f, 0x2, 0xea, 0xeb, 
    0x7, 0x77, 0x2, 0x2, 0xeb, 0xec, 0x7, 0x2e, 0x2, 0x2, 0xec, 0xed, 0x7, 
    0x76, 0x2, 0x2, 0xed, 0xee, 0x5, 0x3a, 0x1e, 0x2, 0xee, 0xef, 0x7, 0x77, 
    0x2, 0x2, 0xef, 0xf0, 0x7, 0x8, 0x2, 0x2, 0xf0, 0x1d, 0x3, 0x2, 0x2, 
    0x2, 0xf1, 0xf6, 0x5, 0x20, 0x11, 0x2, 0xf2, 0xf3, 0x7, 0xd, 0x2, 0x2, 
    0xf3, 0xf5, 0x5, 0x20, 0x11, 0x2, 0xf4, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xf5, 
    0xf8, 0x3, 0x2, 0x2, 0x2, 0xf6, 0xf4, 0x3, 0x2, 0x2, 0x2, 0xf6, 0xf7, 
    0x3, 0x2, 0x2, 0x2, 0xf7, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xf8, 0xf6, 0x3, 
    0x2, 0x2, 0x2, 0xf9, 0xfc, 0x5, 0x2c, 0x17, 0x2, 0xfa, 0xfc, 0x5, 0x2e, 
    0x18, 0x2, 0xfb, 0xf9, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfa, 0x3, 0x2, 0x2, 
    0x2, 0xfc, 0x21, 0x3, 0x2, 0x2, 0x2, 0xfd, 0x102, 0x5, 0x24, 0x13, 0x2, 
    0xfe, 0xff, 0x7, 0xd, 0x2, 0x2, 0xff, 0x101, 0x5, 0x24, 0x13, 0x2, 0x100, 
    0xfe, 0x3, 0x2, 0x2, 0x2, 0x101, 0x104, 0x3, 0x2, 0x2, 0x2, 0x102, 0x100, 
    0x3, 0x2, 0x2, 0x2, 0x102, 0x103, 0x3, 0x2, 0x2, 0x2, 0x103, 0x23, 0x3, 
    0x2, 0x2, 0x2, 0x104, 0x102, 0x3, 0x2, 0x2, 0x2, 0x105, 0x109, 0x5, 
    0x26, 0x14, 0x2, 0x106, 0x109, 0x5, 0x28, 0x15, 0x2, 0x107, 0x109, 0x5, 
    0x2a, 0x16, 0x2, 0x108, 0x105, 0x3, 0x2, 0x2, 0x2, 0x108, 0x106, 0x3, 
    0x2, 0x2, 0x2, 0x108, 0x107, 0x3, 0x2, 0x2, 0x2, 0x109, 0x25, 0x3, 0x2, 
    0x2, 0x2, 0x10a, 0x10b, 0x7, 0x28, 0x2, 0x2, 0x10b, 0x10c, 0x5, 0x52, 
    0x2a, 0x2, 0x10c, 0x10d, 0x7, 0x19, 0x2, 0x2, 0x10d, 0x27, 0x3, 0x2, 
    0x2, 0x2, 0x10e, 0x10f, 0x7, 0x29, 0x2, 0x2, 0x10f, 0x110, 0x5, 0x52, 
    0x2a, 0x2, 0x110, 0x29, 0x3, 0x2, 0x2, 0x2, 0x111, 0x112, 0x7, 0x2a, 
    0x2, 0x2, 0x112, 0x113, 0x5, 0x52, 0x2a, 0x2, 0x113, 0x114, 0x7, 0x19, 
    0x2, 0x2, 0x114, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x115, 0x116, 0x5, 0x52, 
    0x2a, 0x2, 0x116, 0x117, 0x7, 0x19, 0x2, 0x2, 0x117, 0x2d, 0x3, 0x2, 
    0x2, 0x2, 0x118, 0x119, 0x7, 0x2c, 0x2, 0x2, 0x119, 0x11a, 0x5, 0x58, 
    0x2d, 0x2, 0x11a, 0x11b, 0x7, 0x76, 0x2, 0x2, 0x11b, 0x11c, 0x5, 0x3e, 
    0x20, 0x2, 0x11c, 0x11d, 0x7, 0x77, 0x2, 0x2, 0x11d, 0x2f, 0x3, 0x2, 
    0x2, 0x2, 0x11e, 0x123, 0x5, 0x32, 0x1a, 0x2, 0x11f, 0x120, 0x7, 0xd, 
    0x2, 0x2, 0x120, 0x122, 0x5, 0x32, 0x1a, 0x2, 0x121, 0x11f, 0x3, 0x2, 
    0x2, 0x2, 0x122, 0x125, 0x3, 0x2, 0x2, 0x2, 0x123, 0x121, 0x3, 0x2, 
    0x2, 0x2, 0x123, 0x124, 0x3, 0x2, 0x2, 0x2, 0x124, 0x31, 0x3, 0x2, 0x2, 
    0x2, 0x125, 0x123, 0x3, 0x2, 0x2, 0x2, 0x126, 0x129, 0x5, 0x62, 0x32, 
    0x2, 0x127, 0x128, 0x7, 0x34, 0x2, 0x2, 0x128, 0x12a, 0x5, 0x56, 0x2c, 
    0x2, 0x129, 0x127, 0x3, 0x2, 0x2, 0x2, 0x129, 0x12a, 0x3, 0x2, 0x2, 
    0x2, 0x12a, 0x33, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x12c, 0x5, 0x62, 0x32, 
    0x2, 0x12c, 0x35, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x132, 0x5, 0x38, 0x1d, 
    0x2, 0x12e, 0x12f, 0x7, 0xd, 0x2, 0x2, 0x12f, 0x131, 0x5, 0x38, 0x1d, 
    0x2, 0x130, 0x12e, 0x3, 0x2, 0x2, 0x2, 0x131, 0x134, 0x3, 0x2, 0x2, 
    0x2, 0x132, 0x130, 0x3, 0x2, 0x2, 0x2, 0x132, 0x133, 0x3, 0x2, 0x2, 
    0x2, 0x133, 0x37, 0x3, 0x2, 0x2, 0x2, 0x134, 0x132, 0x3, 0x2, 0x2, 0x2, 
    0x135, 0x137, 0x5, 0x44, 0x23, 0x2, 0x136, 0x138, 0x7, 0x39, 0x2, 0x2, 
    0x137, 0x136, 0x3, 0x2, 0x2, 0x2, 0x137, 0x138, 0x3, 0x2, 0x2, 0x2, 
    0x138, 0x39, 0x3, 0x2, 0x2, 0x2, 0x139, 0x13e, 0x5, 0x60, 0x31, 0x2, 
    0x13a, 0x13b, 0x7, 0xd, 0x2, 0x2, 0x13b, 0x13d, 0x5, 0x60, 0x31, 0x2, 
    0x13c, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x13d, 0x140, 0x3, 0x2, 0x2, 0x2, 
    0x13e, 0x13c, 0x3, 0x2, 0x2, 0x2, 0x13e, 0x13f, 0x3, 0x2, 0x2, 0x2, 
    0x13f, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x140, 0x13e, 0x3, 0x2, 0x2, 0x2, 0x141, 
    0x146, 0x5, 0x44, 0x23, 0x2, 0x142, 0x143, 0x7, 0xd, 0x2, 0x2, 0x143, 
    0x145, 0x5, 0x44, 0x23, 0x2, 0x144, 0x142, 0x3, 0x2, 0x2, 0x2, 0x145, 
    0x148, 0x3, 0x2, 0x2, 0x2, 0x146, 0x144, 0x3, 0x2, 0x2, 0x2, 0x146, 
    0x147, 0x3, 0x2, 0x2, 0x2, 0x147, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x148, 0x146, 
    0x3, 0x2, 0x2, 0x2, 0x149, 0x14e, 0x5, 0x52, 0x2a, 0x2, 0x14a, 0x14b, 
    0x7, 0xd, 0x2, 0x2, 0x14b, 0x14d, 0x5, 0x52, 0x2a, 0x2, 0x14c, 0x14a, 
    0x3, 0x2, 0x2, 0x2, 0x14d, 0x150, 0x3, 0x2, 0x2, 0x2, 0x14e, 0x14c, 
    0x3, 0x2, 0x2, 0x2, 0x14e, 0x14f, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x3f, 0x3, 
    0x2, 0x2, 0x2, 0x150, 0x14e, 0x3, 0x2, 0x2, 0x2, 0x151, 0x156, 0x5, 
    0x42, 0x22, 0x2, 0x152, 0x153, 0x7, 0xd, 0x2, 0x2, 0x153, 0x155, 0x5, 
    0x42, 0x22, 0x2, 0x154, 0x152, 0x3, 0x2, 0x2, 0x2, 0x155, 0x158, 0x3, 
    0x2, 0x2, 0x2, 0x156, 0x154, 0x3, 0x2, 0x2, 0x2, 0x156, 0x157, 0x3, 
    0x2, 0x2, 0x2, 0x157, 0x41, 0x3, 0x2, 0x2, 0x2, 0x158, 0x156, 0x3, 0x2, 
    0x2, 0x2, 0x159, 0x15a, 0x5, 0x62, 0x32, 0x2, 0x15a, 0x43, 0x3, 0x2, 
    0x2, 0x2, 0x15b, 0x161, 0x5, 0x52, 0x2a, 0x2, 0x15c, 0x15d, 0x5, 0x50, 
    0x29, 0x2, 0x15d, 0x15e, 0x7, 0xe, 0x2, 0x2, 0x15e, 0x15f, 0x5, 0x52, 
    0x2a, 0x2, 0x15f, 0x161, 0x3, 0x2, 0x2, 0x2, 0x160, 0x15b, 0x3, 0x2, 
    0x2, 0x2, 0x160, 0x15c, 0x3, 0x2, 0x2, 0x2, 0x161, 0x45, 0x3, 0x2, 0x2, 
    0x2, 0x162, 0x167, 0x5, 0x4e, 0x28, 0x2, 0x163, 0x164, 0x7, 0xd, 0x2, 
    0x2, 0x164, 0x166, 0x5, 0x4e, 0x28, 0x2, 0x165, 0x163, 0x3, 0x2, 0x2, 
    0x2, 0x166, 0x169, 0x3, 0x2, 0x2, 0x2, 0x167, 0x165, 0x3, 0x2, 0x2, 
    0x2, 0x167, 0x168, 0x3, 0x2, 0x2, 0x2, 0x168, 0x47, 0x3, 0x2, 0x2, 0x2, 
    0x169, 0x167, 0x3, 0x2, 0x2, 0x2, 0x16a, 0x16c, 0x5, 0x4a, 0x26, 0x2, 
    0x16b, 0x16a, 0x3, 0x2, 0x2, 0x2, 0x16c, 0x16d, 0x3, 0x2, 0x2, 0x2, 
    0x16d, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x16d, 0x16e, 0x3, 0x2, 0x2, 0x2, 
    0x16e, 0x49, 0x3, 0x2, 0x2, 0x2, 0x16f, 0x170, 0x7, 0x31, 0x2, 0x2, 
    0x170, 0x171, 0x5, 0x4c, 0x27, 0x2, 0x171, 0x172, 0x7, 0x37, 0x2, 0x2, 
    0x172, 0x173, 0x5, 0x62, 0x32, 0x2, 0x173, 0x4b, 0x3, 0x2, 0x2, 0x2, 
    0x174, 0x177, 0x5, 0x50, 0x29, 0x2, 0x175, 0x176, 0x7, 0x34, 0x2, 0x2, 
    0x176, 0x178, 0x5, 0x56, 0x2c, 0x2, 0x177, 0x175, 0x3, 0x2, 0x2, 0x2, 
    0x177, 0x178, 0x3, 0x2, 0x2, 0x2, 0x178, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x179, 
    0x17c, 0x5, 0x50, 0x29, 0x2, 0x17a, 0x17b, 0x7, 0x34, 0x2, 0x2, 0x17b, 
    0x17d, 0x5, 0x56, 0x2c, 0x2, 0x17c, 0x17a, 0x3, 0x2, 0x2, 0x2, 0x17c, 
    0x17d, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x17e, 0x17f, 
    0x9, 0x3, 0x2, 0x2, 0x17f, 0x51, 0x3, 0x2, 0x2, 0x2, 0x180, 0x181, 0x9, 
    0x3, 0x2, 0x2, 0x181, 0x53, 0x3, 0x2, 0x2, 0x2, 0x182, 0x183, 0x9, 0x3, 
    0x2, 0x2, 0x183, 0x55, 0x3, 0x2, 0x2, 0x2, 0x184, 0x185, 0x9, 0x3, 0x2, 
    0x2, 0x185, 0x57, 0x3, 0x2, 0x2, 0x2, 0x186, 0x187, 0x9, 0x3, 0x2, 0x2, 
    0x187, 0x59, 0x3, 0x2, 0x2, 0x2, 0x188, 0x189, 0x7, 0x87, 0x2, 0x2, 
    0x189, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x18a, 0x18b, 0x7, 0x87, 0x2, 0x2, 
    0x18b, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x18c, 0x18d, 0x7, 0x87, 0x2, 0x2, 
    0x18d, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x18e, 0x193, 0x7, 0x87, 0x2, 0x2, 
    0x18f, 0x193, 0x7, 0x86, 0x2, 0x2, 0x190, 0x193, 0x5, 0x64, 0x33, 0x2, 
    0x191, 0x193, 0x7, 0xf, 0x2, 0x2, 0x192, 0x18e, 0x3, 0x2, 0x2, 0x2, 
    0x192, 0x18f, 0x3, 0x2, 0x2, 0x2, 0x192, 0x190, 0x3, 0x2, 0x2, 0x2, 
    0x192, 0x191, 0x3, 0x2, 0x2, 0x2, 0x193, 0x61, 0x3, 0x2, 0x2, 0x2, 0x194, 
    0x195, 0x8, 0x32, 0x1, 0x2, 0x195, 0x196, 0x7, 0x7c, 0x2, 0x2, 0x196, 
    0x28a, 0x5, 0x62, 0x32, 0x44, 0x197, 0x198, 0x7, 0x6e, 0x2, 0x2, 0x198, 
    0x28a, 0x5, 0x62, 0x32, 0x43, 0x199, 0x19a, 0x7, 0x4d, 0x2, 0x2, 0x19a, 
    0x19b, 0x7, 0x76, 0x2, 0x2, 0x19b, 0x19c, 0x5, 0x62, 0x32, 0x2, 0x19c, 
    0x19d, 0x7, 0x77, 0x2, 0x2, 0x19d, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x19e, 
    0x19f, 0x7, 0x4e, 0x2, 0x2, 0x19f, 0x1a0, 0x7, 0x76, 0x2, 0x2, 0x1a0, 
    0x1a1, 0x5, 0x62, 0x32, 0x2, 0x1a1, 0x1a2, 0x7, 0x77, 0x2, 0x2, 0x1a2, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1a3, 0x1a4, 0x7, 0x4f, 0x2, 0x2, 0x1a4, 
    0x1a5, 0x7, 0x76, 0x2, 0x2, 0x1a5, 0x1a6, 0x5, 0x62, 0x32, 0x2, 0x1a6, 
    0x1a7, 0x7, 0x77, 0x2, 0x2, 0x1a7, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1a8, 
    0x1a9, 0x7, 0x50, 0x2, 0x2, 0x1a9, 0x1aa, 0x7, 0x76, 0x2, 0x2, 0x1aa, 
    0x1ab, 0x5, 0x62, 0x32, 0x2, 0x1ab, 0x1ac, 0x7, 0x77, 0x2, 0x2, 0x1ac, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1ad, 0x1ae, 0x7, 0x51, 0x2, 0x2, 0x1ae, 
    0x1af, 0x7, 0x76, 0x2, 0x2, 0x1af, 0x1b0, 0x5, 0x62, 0x32, 0x2, 0x1b0, 
    0x1b1, 0x7, 0x77, 0x2, 0x2, 0x1b1, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1b2, 
    0x1b3, 0x7, 0x52, 0x2, 0x2, 0x1b3, 0x1b4, 0x7, 0x76, 0x2, 0x2, 0x1b4, 
    0x1b5, 0x5, 0x62, 0x32, 0x2, 0x1b5, 0x1b6, 0x7, 0x77, 0x2, 0x2, 0x1b6, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1b7, 0x1b8, 0x7, 0x53, 0x2, 0x2, 0x1b8, 
    0x1b9, 0x7, 0x76, 0x2, 0x2, 0x1b9, 0x1ba, 0x5, 0x62, 0x32, 0x2, 0x1ba, 
    0x1bb, 0x7, 0x77, 0x2, 0x2, 0x1bb, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1bc, 
    0x1bd, 0x7, 0x54, 0x2, 0x2, 0x1bd, 0x1be, 0x7, 0x76, 0x2, 0x2, 0x1be, 
    0x1bf, 0x5, 0x62, 0x32, 0x2, 0x1bf, 0x1c0, 0x7, 0x77, 0x2, 0x2, 0x1c0, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1c1, 0x1c2, 0x7, 0x56, 0x2, 0x2, 0x1c2, 
    0x1c3, 0x7, 0x76, 0x2, 0x2, 0x1c3, 0x1c4, 0x5, 0x62, 0x32, 0x2, 0x1c4, 
    0x1c5, 0x7, 0x77, 0x2, 0x2, 0x1c5, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1c6, 
    0x1c7, 0x7, 0x57, 0x2, 0x2, 0x1c7, 0x1c8, 0x7, 0x76, 0x2, 0x2, 0x1c8, 
    0x1c9, 0x5, 0x62, 0x32, 0x2, 0x1c9, 0x1ca, 0x7, 0x77, 0x2, 0x2, 0x1ca, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1cb, 0x1cc, 0x7, 0x58, 0x2, 0x2, 0x1cc, 
    0x1cd, 0x7, 0x76, 0x2, 0x2, 0x1cd, 0x1ce, 0x5, 0x62, 0x32, 0x2, 0x1ce, 
    0x1cf, 0x7, 0x77, 0x2, 0x2, 0x1cf, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1d0, 
    0x1d1, 0x7, 0x5a, 0x2, 0x2, 0x1d1, 0x1d2, 0x7, 0x76, 0x2, 0x2, 0x1d2, 
    0x1d3, 0x5, 0x62, 0x32, 0x2, 0x1d3, 0x1d4, 0x7, 0x77, 0x2, 0x2, 0x1d4, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1d5, 0x1d6, 0x7, 0x5b, 0x2, 0x2, 0x1d6, 
    0x1d7, 0x7, 0x76, 0x2, 0x2, 0x1d7, 0x1d8, 0x5, 0x62, 0x32, 0x2, 0x1d8, 
    0x1d9, 0x7, 0x77, 0x2, 0x2, 0x1d9, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1da, 
    0x1db, 0x7, 0x5c, 0x2, 0x2, 0x1db, 0x1dc, 0x7, 0x76, 0x2, 0x2, 0x1dc, 
    0x1dd, 0x5, 0x62, 0x32, 0x2, 0x1dd, 0x1de, 0x7, 0x77, 0x2, 0x2, 0x1de, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1df, 0x1e0, 0x7, 0x5e, 0x2, 0x2, 0x1e0, 
    0x1e1, 0x7, 0x76, 0x2, 0x2, 0x1e1, 0x1e2, 0x5, 0x62, 0x32, 0x2, 0x1e2, 
    0x1e3, 0x7, 0x77, 0x2, 0x2, 0x1e3, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1e4, 
    0x1e5, 0x7, 0x60, 0x2, 0x2, 0x1e5, 0x1e6, 0x7, 0x76, 0x2, 0x2, 0x1e6, 
    0x1e7, 0x5, 0x62, 0x32, 0x2, 0x1e7, 0x1e8, 0x7, 0x77, 0x2, 0x2, 0x1e8, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1e9, 0x1ea, 0x7, 0x5f, 0x2, 0x2, 0x1ea, 
    0x1eb, 0x7, 0x76, 0x2, 0x2, 0x1eb, 0x1ec, 0x5, 0x62, 0x32, 0x2, 0x1ec, 
    0x1ed, 0x7, 0x77, 0x2, 0x2, 0x1ed, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1ee, 
    0x1ef, 0x7, 0x45, 0x2, 0x2, 0x1ef, 0x1f0, 0x7, 0x76, 0x2, 0x2, 0x1f0, 
    0x1f1, 0x5, 0x62, 0x32, 0x2, 0x1f1, 0x1f2, 0x7, 0x77, 0x2, 0x2, 0x1f2, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1f3, 0x1f4, 0x7, 0x46, 0x2, 0x2, 0x1f4, 
    0x1f5, 0x7, 0x76, 0x2, 0x2, 0x1f5, 0x1f6, 0x5, 0x62, 0x32, 0x2, 0x1f6, 
    0x1f7, 0x7, 0x77, 0x2, 0x2, 0x1f7, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x1f8, 
    0x1f9, 0x7, 0x47, 0x2, 0x2, 0x1f9, 0x1fa, 0x7, 0x76, 0x2, 0x2, 0x1fa, 
    0x1fb, 0x5, 0x62, 0x32, 0x2, 0x1fb, 0x1fc, 0x7, 0x77, 0x2, 0x2, 0x1fc, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x1fd, 0x1fe, 0x7, 0x48, 0x2, 0x2, 0x1fe, 
    0x1ff, 0x7, 0x76, 0x2, 0x2, 0x1ff, 0x200, 0x5, 0x62, 0x32, 0x2, 0x200, 
    0x201, 0x7, 0x77, 0x2, 0x2, 0x201, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x202, 
    0x203, 0x7, 0x49, 0x2, 0x2, 0x203, 0x204, 0x7, 0x76, 0x2, 0x2, 0x204, 
    0x205, 0x5, 0x62, 0x32, 0x2, 0x205, 0x206, 0x7, 0x77, 0x2, 0x2, 0x206, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x207, 0x208, 0x7, 0x4a, 0x2, 0x2, 0x208, 
    0x209, 0x7, 0x76, 0x2, 0x2, 0x209, 0x20a, 0x5, 0x62, 0x32, 0x2, 0x20a, 
    0x20b, 0x7, 0x77, 0x2, 0x2, 0x20b, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x20c, 
    0x20d, 0x7, 0x61, 0x2, 0x2, 0x20d, 0x20e, 0x7, 0x76, 0x2, 0x2, 0x20e, 
    0x20f, 0x5, 0x62, 0x32, 0x2, 0x20f, 0x210, 0x7, 0x77, 0x2, 0x2, 0x210, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x211, 0x212, 0x7, 0x62, 0x2, 0x2, 0x212, 
    0x213, 0x7, 0x76, 0x2, 0x2, 0x213, 0x214, 0x5, 0x62, 0x32, 0x2, 0x214, 
    0x215, 0x7, 0x77, 0x2, 0x2, 0x215, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x216, 
    0x217, 0x7, 0x63, 0x2, 0x2, 0x217, 0x218, 0x7, 0x76, 0x2, 0x2, 0x218, 
    0x219, 0x5, 0x62, 0x32, 0x2, 0x219, 0x21a, 0x7, 0x77, 0x2, 0x2, 0x21a, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x21b, 0x21c, 0x7, 0x64, 0x2, 0x2, 0x21c, 
    0x21d, 0x7, 0x76, 0x2, 0x2, 0x21d, 0x21e, 0x5, 0x62, 0x32, 0x2, 0x21e, 
    0x21f, 0x7, 0x77, 0x2, 0x2, 0x21f, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x220, 
    0x221, 0x7, 0x65, 0x2, 0x2, 0x221, 0x222, 0x7, 0x76, 0x2, 0x2, 0x222, 
    0x223, 0x5, 0x62, 0x32, 0x2, 0x223, 0x224, 0x7, 0x77, 0x2, 0x2, 0x224, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x225, 0x226, 0x7, 0x66, 0x2, 0x2, 0x226, 
    0x227, 0x7, 0x76, 0x2, 0x2, 0x227, 0x228, 0x5, 0x62, 0x32, 0x2, 0x228, 
    0x229, 0x7, 0x77, 0x2, 0x2, 0x229, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x22a, 
    0x22b, 0x7, 0x55, 0x2, 0x2, 0x22b, 0x22c, 0x7, 0x76, 0x2, 0x2, 0x22c, 
    0x22d, 0x5, 0x62, 0x32, 0x2, 0x22d, 0x22e, 0x7, 0xd, 0x2, 0x2, 0x22e, 
    0x22f, 0x5, 0x62, 0x32, 0x2, 0x22f, 0x230, 0x7, 0x77, 0x2, 0x2, 0x230, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x231, 0x232, 0x7, 0x57, 0x2, 0x2, 0x232, 
    0x233, 0x7, 0x76, 0x2, 0x2, 0x233, 0x234, 0x5, 0x62, 0x32, 0x2, 0x234, 
    0x235, 0x7, 0xd, 0x2, 0x2, 0x235, 0x236, 0x5, 0x62, 0x32, 0x2, 0x236, 
    0x237, 0x7, 0x77, 0x2, 0x2, 0x237, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x238, 
    0x239, 0x7, 0x59, 0x2, 0x2, 0x239, 0x23a, 0x7, 0x76, 0x2, 0x2, 0x23a, 
    0x23b, 0x5, 0x62, 0x32, 0x2, 0x23b, 0x23c, 0x7, 0xd, 0x2, 0x2, 0x23c, 
    0x23d, 0x5, 0x62, 0x32, 0x2, 0x23d, 0x23e, 0x7, 0x77, 0x2, 0x2, 0x23e, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x23f, 0x240, 0x7, 0x5d, 0x2, 0x2, 0x240, 
    0x241, 0x7, 0x76, 0x2, 0x2, 0x241, 0x242, 0x5, 0x62, 0x32, 0x2, 0x242, 
    0x243, 0x7, 0xd, 0x2, 0x2, 0x243, 0x244, 0x5, 0x62, 0x32, 0x2, 0x244, 
    0x245, 0x7, 0x77, 0x2, 0x2, 0x245, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x246, 
    0x247, 0x7, 0x13, 0x2, 0x2, 0x247, 0x248, 0x7, 0x76, 0x2, 0x2, 0x248, 
    0x249, 0x5, 0x62, 0x32, 0x2, 0x249, 0x24a, 0x7, 0xd, 0x2, 0x2, 0x24a, 
    0x24b, 0x5, 0x62, 0x32, 0x2, 0x24b, 0x24c, 0x7, 0x77, 0x2, 0x2, 0x24c, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x24d, 0x24e, 0x7, 0x6a, 0x2, 0x2, 0x24e, 
    0x24f, 0x7, 0x76, 0x2, 0x2, 0x24f, 0x250, 0x5, 0x62, 0x32, 0x2, 0x250, 
    0x251, 0x7, 0xd, 0x2, 0x2, 0x251, 0x252, 0x5, 0x62, 0x32, 0x2, 0x252, 
    0x253, 0x7, 0x77, 0x2, 0x2, 0x253, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x254, 
    0x255, 0x7, 0x6b, 0x2, 0x2, 0x255, 0x256, 0x7, 0x76, 0x2, 0x2, 0x256, 
    0x257, 0x5, 0x62, 0x32, 0x2, 0x257, 0x258, 0x7, 0xd, 0x2, 0x2, 0x258, 
    0x259, 0x5, 0x62, 0x32, 0x2, 0x259, 0x25a, 0x7, 0x77, 0x2, 0x2, 0x25a, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x25b, 0x25c, 0x7, 0x6c, 0x2, 0x2, 0x25c, 
    0x25d, 0x7, 0x76, 0x2, 0x2, 0x25d, 0x25e, 0x5, 0x62, 0x32, 0x2, 0x25e, 
    0x25f, 0x7, 0xd, 0x2, 0x2, 0x25f, 0x260, 0x5, 0x62, 0x32, 0x2, 0x260, 
    0x261, 0x7, 0x77, 0x2, 0x2, 0x261, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x262, 
    0x263, 0x7, 0x69, 0x2, 0x2, 0x263, 0x264, 0x7, 0x76, 0x2, 0x2, 0x264, 
    0x265, 0x5, 0x62, 0x32, 0x2, 0x265, 0x266, 0x7, 0xd, 0x2, 0x2, 0x266, 
    0x267, 0x5, 0x62, 0x32, 0x2, 0x267, 0x268, 0x7, 0x77, 0x2, 0x2, 0x268, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x269, 0x26a, 0x7, 0x67, 0x2, 0x2, 0x26a, 
    0x26b, 0x7, 0x76, 0x2, 0x2, 0x26b, 0x26c, 0x5, 0x62, 0x32, 0x2, 0x26c, 
    0x26d, 0x7, 0xd, 0x2, 0x2, 0x26d, 0x26e, 0x5, 0x62, 0x32, 0x2, 0x26e, 
    0x26f, 0x7, 0x77, 0x2, 0x2, 0x26f, 0x28a, 0x3, 0x2, 0x2, 0x2, 0x270, 
    0x271, 0x7, 0x68, 0x2, 0x2, 0x271, 0x272, 0x7, 0x76, 0x2, 0x2, 0x272, 
    0x273, 0x5, 0x62, 0x32, 0x2, 0x273, 0x274, 0x7, 0xd, 0x2, 0x2, 0x274, 
    0x275, 0x5, 0x62, 0x32, 0x2, 0x275, 0x276, 0x7, 0x77, 0x2, 0x2, 0x276, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x277, 0x278, 0x7, 0x76, 0x2, 0x2, 0x278, 
    0x279, 0x5, 0x62, 0x32, 0x2, 0x279, 0x27a, 0x7, 0x77, 0x2, 0x2, 0x27a, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x28a, 0x5, 0x44, 0x23, 0x2, 0x27c, 
    0x28a, 0x5, 0x64, 0x33, 0x2, 0x27d, 0x28a, 0x7, 0x3, 0x2, 0x2, 0x27e, 
    0x28a, 0x7, 0x86, 0x2, 0x2, 0x27f, 0x28a, 0x7, 0x4c, 0x2, 0x2, 0x280, 
    0x28a, 0x7, 0x4b, 0x2, 0x2, 0x281, 0x28a, 0x7, 0x87, 0x2, 0x2, 0x282, 
    0x28a, 0x7, 0xf, 0x2, 0x2, 0x283, 0x28a, 0x7, 0x83, 0x2, 0x2, 0x284, 
    0x285, 0x7, 0x3f, 0x2, 0x2, 0x285, 0x286, 0x7, 0x76, 0x2, 0x2, 0x286, 
    0x287, 0x5, 0x62, 0x32, 0x2, 0x287, 0x288, 0x7, 0x77, 0x2, 0x2, 0x288, 
    0x28a, 0x3, 0x2, 0x2, 0x2, 0x289, 0x194, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x197, 0x3, 0x2, 0x2, 0x2, 0x289, 0x199, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x19e, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1a3, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1a8, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1ad, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1b2, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1b7, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1bc, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1c1, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1c6, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1cb, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1d0, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1d5, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1da, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1df, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1e4, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1e9, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1ee, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1f3, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x1f8, 0x3, 0x2, 0x2, 0x2, 0x289, 0x1fd, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x202, 0x3, 0x2, 0x2, 0x2, 0x289, 0x207, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x20c, 0x3, 0x2, 0x2, 0x2, 0x289, 0x211, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x216, 0x3, 0x2, 0x2, 0x2, 0x289, 0x21b, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x220, 0x3, 0x2, 0x2, 0x2, 0x289, 0x225, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x22a, 0x3, 0x2, 0x2, 0x2, 0x289, 0x231, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x238, 0x3, 0x2, 0x2, 0x2, 0x289, 0x23f, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x246, 0x3, 0x2, 0x2, 0x2, 0x289, 0x24d, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x254, 0x3, 0x2, 0x2, 0x2, 0x289, 0x25b, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x262, 0x3, 0x2, 0x2, 0x2, 0x289, 0x269, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x270, 0x3, 0x2, 0x2, 0x2, 0x289, 0x277, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x27b, 0x3, 0x2, 0x2, 0x2, 0x289, 0x27c, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x27d, 0x3, 0x2, 0x2, 0x2, 0x289, 0x27e, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x27f, 0x3, 0x2, 0x2, 0x2, 0x289, 0x280, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x281, 0x3, 0x2, 0x2, 0x2, 0x289, 0x282, 0x3, 0x2, 0x2, 0x2, 0x289, 
    0x283, 0x3, 0x2, 0x2, 0x2, 0x289, 0x284, 0x3, 0x2, 0x2, 0x2, 0x28a, 
    0x2b7, 0x3, 0x2, 0x2, 0x2, 0x28b, 0x28c, 0xc, 0x25, 0x2, 0x2, 0x28c, 
    0x28d, 0x9, 0x4, 0x2, 0x2, 0x28d, 0x2b6, 0x5, 0x62, 0x32, 0x26, 0x28e, 
    0x28f, 0xc, 0x24, 0x2, 0x2, 0x28f, 0x290, 0x9, 0x5, 0x2, 0x2, 0x290, 
    0x2b6, 0x5, 0x62, 0x32, 0x25, 0x291, 0x292, 0xc, 0x23, 0x2, 0x2, 0x292, 
    0x293, 0x7, 0x71, 0x2, 0x2, 0x293, 0x2b6, 0x5, 0x62, 0x32, 0x24, 0x294, 
    0x295, 0xc, 0x1e, 0x2, 0x2, 0x295, 0x296, 0x7, 0x72, 0x2, 0x2, 0x296, 
    0x2b6, 0x5, 0x62, 0x32, 0x1f, 0x297, 0x298, 0xc, 0x1d, 0x2, 0x2, 0x298, 
    0x299, 0x9, 0x6, 0x2, 0x2, 0x299, 0x2b6, 0x5, 0x62, 0x32, 0x1e, 0x29a, 
    0x29b, 0xc, 0x1c, 0x2, 0x2, 0x29b, 0x29c, 0x9, 0x7, 0x2, 0x2, 0x29c, 
    0x2b6, 0x5, 0x62, 0x32, 0x1d, 0x29d, 0x29e, 0xc, 0x1b, 0x2, 0x2, 0x29e, 
    0x29f, 0x9, 0x8, 0x2, 0x2, 0x29f, 0x2b6, 0x5, 0x62, 0x32, 0x1c, 0x2a0, 
    0x2a1, 0xc, 0x1a, 0x2, 0x2, 0x2a1, 0x2a2, 0x9, 0x9, 0x2, 0x2, 0x2a2, 
    0x2b6, 0x5, 0x62, 0x32, 0x1b, 0x2a3, 0x2a4, 0xc, 0x19, 0x2, 0x2, 0x2a4, 
    0x2a5, 0x9, 0xa, 0x2, 0x2, 0x2a5, 0x2b6, 0x5, 0x62, 0x32, 0x1a, 0x2a6, 
    0x2a7, 0xc, 0x18, 0x2, 0x2, 0x2a7, 0x2a8, 0x7, 0x75, 0x2, 0x2, 0x2a8, 
    0x2b6, 0x5, 0x62, 0x32, 0x19, 0x2a9, 0x2aa, 0xc, 0x10, 0x2, 0x2, 0x2aa, 
    0x2ab, 0x7, 0x36, 0x2, 0x2, 0x2ab, 0x2ac, 0x5, 0x62, 0x32, 0x2, 0x2ac, 
    0x2ad, 0x7, 0x7e, 0x2, 0x2, 0x2ad, 0x2ae, 0x5, 0x62, 0x32, 0x11, 0x2ae, 
    0x2b6, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x2b0, 0xc, 0xf, 0x2, 0x2, 0x2b0, 
    0x2b1, 0x7, 0x7e, 0x2, 0x2, 0x2b1, 0x2b6, 0x5, 0x62, 0x32, 0x10, 0x2b2, 
    0x2b3, 0xc, 0xe, 0x2, 0x2, 0x2b3, 0x2b4, 0x7, 0x7d, 0x2, 0x2, 0x2b4, 
    0x2b6, 0x5, 0x62, 0x32, 0xf, 0x2b5, 0x28b, 0x3, 0x2, 0x2, 0x2, 0x2b5, 
    0x28e, 0x3, 0x2, 0x2, 0x2, 0x2b5, 0x291, 0x3, 0x2, 0x2, 0x2, 0x2b5, 
    0x294, 0x3, 0x2, 0x2, 0x2, 0x2b5, 0x297, 0x3, 0x2, 0x2, 0x2, 0x2b5, 
    0x29a, 0x3, 0x2, 0x2, 0x2, 0x2b5, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x2b5, 
    0x2a0, 0x3, 0x2, 0x2, 0x2, 0x2b5, 0x2a3, 0x3, 0x2, 0x2, 0x2, 0x2b5, 
    0x2a6, 0x3, 0x2, 0x2, 0x2, 0x2b5, 0x2a9, 0x3, 0x2, 0x2, 0x2, 0x2b5, 
    0x2af, 0x3, 0x2, 0x2, 0x2, 0x2b5, 0x2b2, 0x3, 0x2, 0x2, 0x2, 0x2b6, 
    0x2b9, 0x3, 0x2, 0x2, 0x2, 0x2b7, 0x2b5, 0x3, 0x2, 0x2, 0x2, 0x2b7, 
    0x2b8, 0x3, 0x2, 0x2, 0x2, 0x2b8, 0x63, 0x3, 0x2, 0x2, 0x2, 0x2b9, 0x2b7, 
    0x3, 0x2, 0x2, 0x2, 0x2ba, 0x2c1, 0x5, 0x66, 0x34, 0x2, 0x2bb, 0x2c1, 
    0x5, 0x6a, 0x36, 0x2, 0x2bc, 0x2c1, 0x5, 0x68, 0x35, 0x2, 0x2bd, 0x2c1, 
    0x5, 0x6c, 0x37, 0x2, 0x2be, 0x2c1, 0x5, 0x6e, 0x38, 0x2, 0x2bf, 0x2c1, 
    0x5, 0x70, 0x39, 0x2, 0x2c0, 0x2ba, 0x3, 0x2, 0x2, 0x2, 0x2c0, 0x2bb, 
    0x3, 0x2, 0x2, 0x2, 0x2c0, 0x2bc, 0x3, 0x2, 0x2, 0x2, 0x2c0, 0x2bd, 
    0x3, 0x2, 0x2, 0x2, 0x2c0, 0x2be, 0x3, 0x2, 0x2, 0x2, 0x2c0, 0x2bf, 
    0x3, 0x2, 0x2, 0x2, 0x2c1, 0x65, 0x3, 0x2, 0x2, 0x2, 0x2c2, 0x2c3, 0x7, 
    0x13, 0x2, 0x2, 0x2c3, 0x2c4, 0x7, 0x76, 0x2, 0x2, 0x2c4, 0x2c5, 0x5, 
    0x78, 0x3d, 0x2, 0x2c5, 0x2c6, 0x7, 0x77, 0x2, 0x2, 0x2c6, 0x67, 0x3, 
    0x2, 0x2, 0x2, 0x2c7, 0x2c8, 0x7, 0x15, 0x2, 0x2, 0x2c8, 0x2c9, 0x5, 
    0x76, 0x3c, 0x2, 0x2c9, 0x69, 0x3, 0x2, 0x2, 0x2, 0x2ca, 0x2cb, 0x7, 
    0x17, 0x2, 0x2, 0x2cb, 0x2cc, 0x5, 0x74, 0x3b, 0x2, 0x2cc, 0x6b, 0x3, 
    0x2, 0x2, 0x2, 0x2cd, 0x2ce, 0x7, 0x14, 0x2, 0x2, 0x2ce, 0x2cf, 0x7, 
    0x76, 0x2, 0x2, 0x2cf, 0x2d4, 0x5, 0x72, 0x3a, 0x2, 0x2d0, 0x2d1, 0x7, 
    0xd, 0x2, 0x2, 0x2d1, 0x2d3, 0x5, 0x72, 0x3a, 0x2, 0x2d2, 0x2d0, 0x3, 
    0x2, 0x2, 0x2, 0x2d3, 0x2d6, 0x3, 0x2, 0x2, 0x2, 0x2d4, 0x2d2, 0x3, 
    0x2, 0x2, 0x2, 0x2d4, 0x2d5, 0x3, 0x2, 0x2, 0x2, 0x2d5, 0x2d7, 0x3, 
    0x2, 0x2, 0x2, 0x2d6, 0x2d4, 0x3, 0x2, 0x2, 0x2, 0x2d7, 0x2d8, 0x7, 
    0x77, 0x2, 0x2, 0x2d8, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x2d9, 0x2da, 0x7, 
    0x16, 0x2, 0x2, 0x2da, 0x2db, 0x7, 0x76, 0x2, 0x2, 0x2db, 0x2e0, 0x5, 
    0x76, 0x3c, 0x2, 0x2dc, 0x2dd, 0x7, 0xd, 0x2, 0x2, 0x2dd, 0x2df, 0x5, 
    0x76, 0x3c, 0x2, 0x2de, 0x2dc, 0x3, 0x2, 0x2, 0x2, 0x2df, 0x2e2, 0x3, 
    0x2, 0x2, 0x2, 0x2e0, 0x2de, 0x3, 0x2, 0x2, 0x2, 0x2e0, 0x2e1, 0x3, 
    0x2, 0x2, 0x2, 0x2e1, 0x2e3, 0x3, 0x2, 0x2, 0x2, 0x2e2, 0x2e0, 0x3, 
    0x2, 0x2, 0x2, 0x2e3, 0x2e4, 0x7, 0x77, 0x2, 0x2, 0x2e4, 0x6f, 0x3, 
    0x2, 0x2, 0x2, 0x2e5, 0x2e6, 0x7, 0x18, 0x2, 0x2, 0x2e6, 0x2e7, 0x7, 
    0x76, 0x2, 0x2, 0x2e7, 0x2ec, 0x5, 0x74, 0x3b, 0x2, 0x2e8, 0x2e9, 0x7, 
    0xd, 0x2, 0x2, 0x2e9, 0x2eb, 0x5, 0x74, 0x3b, 0x2, 0x2ea, 0x2e8, 0x3, 
    0x2, 0x2, 0x2, 0x2eb, 0x2ee, 0x3, 0x2, 0x2, 0x2, 0x2ec, 0x2ea, 0x3, 
    0x2, 0x2, 0x2, 0x2ec, 0x2ed, 0x3, 0x2, 0x2, 0x2, 0x2ed, 0x2ef, 0x3, 
    0x2, 0x2, 0x2, 0x2ee, 0x2ec, 0x3, 0x2, 0x2, 0x2, 0x2ef, 0x2f0, 0x7, 
    0x77, 0x2, 0x2, 0x2f0, 0x71, 0x3, 0x2, 0x2, 0x2, 0x2f1, 0x2f7, 0x5, 
    0x78, 0x3d, 0x2, 0x2f2, 0x2f3, 0x7, 0x76, 0x2, 0x2, 0x2f3, 0x2f4, 0x5, 
    0x78, 0x3d, 0x2, 0x2f4, 0x2f5, 0x7, 0x77, 0x2, 0x2, 0x2f5, 0x2f7, 0x3, 
    0x2, 0x2, 0x2, 0x2f6, 0x2f1, 0x3, 0x2, 0x2, 0x2, 0x2f6, 0x2f2, 0x3, 
    0x2, 0x2, 0x2, 0x2f7, 0x73, 0x3, 0x2, 0x2, 0x2, 0x2f8, 0x2f9, 0x7, 0x76, 
    0x2, 0x2, 0x2f9, 0x2fe, 0x5, 0x76, 0x3c, 0x2, 0x2fa, 0x2fb, 0x7, 0xd, 
    0x2, 0x2, 0x2fb, 0x2fd, 0x5, 0x76, 0x3c, 0x2, 0x2fc, 0x2fa, 0x3, 0x2, 
    0x2, 0x2, 0x2fd, 0x300, 0x3, 0x2, 0x2, 0x2, 0x2fe, 0x2fc, 0x3, 0x2, 
    0x2, 0x2, 0x2fe, 0x2ff, 0x3, 0x2, 0x2, 0x2, 0x2ff, 0x301, 0x3, 0x2, 
    0x2, 0x2, 0x300, 0x2fe, 0x3, 0x2, 0x2, 0x2, 0x301, 0x302, 0x7, 0x77, 
    0x2, 0x2, 0x302, 0x75, 0x3, 0x2, 0x2, 0x2, 0x303, 0x304, 0x7, 0x76, 
    0x2, 0x2, 0x304, 0x309, 0x5, 0x78, 0x3d, 0x2, 0x305, 0x306, 0x7, 0xd, 
    0x2, 0x2, 0x306, 0x308, 0x5, 0x78, 0x3d, 0x2, 0x307, 0x305, 0x3, 0x2, 
    0x2, 0x2, 0x308, 0x30b, 0x3, 0x2, 0x2, 0x2, 0x309, 0x307, 0x3, 0x2, 
    0x2, 0x2, 0x309, 0x30a, 0x3, 0x2, 0x2, 0x2, 0x30a, 0x30c, 0x3, 0x2, 
    0x2, 0x2, 0x30b, 0x309, 0x3, 0x2, 0x2, 0x2, 0x30c, 0x30d, 0x7, 0x77, 
    0x2, 0x2, 0x30d, 0x77, 0x3, 0x2, 0x2, 0x2, 0x30e, 0x30f, 0x9, 0xb, 0x2, 
    0x2, 0x30f, 0x310, 0x9, 0xb, 0x2, 0x2, 0x310, 0x79, 0x3, 0x2, 0x2, 0x2, 
    0x2a, 0x7d, 0x8b, 0x90, 0x98, 0xa1, 0xaa, 0xae, 0xb2, 0xb6, 0xba, 0xbe, 
    0xc5, 0xf6, 0xfb, 0x102, 0x108, 0x123, 0x129, 0x132, 0x137, 0x13e, 0x146, 
    0x14e, 0x156, 0x160, 0x167, 0x16d, 0x177, 0x17c, 0x192, 0x289, 0x2b5, 
    0x2b7, 0x2c0, 0x2d4, 0x2e0, 0x2ec, 0x2f6, 0x2fe, 0x309, 
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

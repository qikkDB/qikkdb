
// Generated from C:/Users/mstano/dropdbase_instarea/dropdbase/GpuSqlParser\GpuSqlParser.g4 by ANTLR 4.7.2


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
    setState(121);
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
      setState(118);
      statement();
      setState(123);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(124);
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
    setState(135);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SELECT: {
        enterOuterAlt(_localctx, 1);
        setState(126);
        sqlSelect();
        break;
      }

      case GpuSqlParser::CREATEDB: {
        enterOuterAlt(_localctx, 2);
        setState(127);
        sqlCreateDb();
        break;
      }

      case GpuSqlParser::DROPDB: {
        enterOuterAlt(_localctx, 3);
        setState(128);
        sqlDropDb();
        break;
      }

      case GpuSqlParser::CREATETABLE: {
        enterOuterAlt(_localctx, 4);
        setState(129);
        sqlCreateTable();
        break;
      }

      case GpuSqlParser::DROPTABLE: {
        enterOuterAlt(_localctx, 5);
        setState(130);
        sqlDropTable();
        break;
      }

      case GpuSqlParser::ALTERTABLE: {
        enterOuterAlt(_localctx, 6);
        setState(131);
        sqlAlterTable();
        break;
      }

      case GpuSqlParser::CREATEINDEX: {
        enterOuterAlt(_localctx, 7);
        setState(132);
        sqlCreateIndex();
        break;
      }

      case GpuSqlParser::INSERTINTO: {
        enterOuterAlt(_localctx, 8);
        setState(133);
        sqlInsertInto();
        break;
      }

      case GpuSqlParser::SHOWDB:
      case GpuSqlParser::SHOWTB:
      case GpuSqlParser::SHOWCL: {
        enterOuterAlt(_localctx, 9);
        setState(134);
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
    setState(140);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SHOWDB: {
        setState(137);
        showDatabases();
        break;
      }

      case GpuSqlParser::SHOWTB: {
        setState(138);
        showTables();
        break;
      }

      case GpuSqlParser::SHOWCL: {
        setState(139);
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
    setState(142);
    match(GpuSqlParser::SHOWDB);
    setState(143);
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
    setState(145);
    match(GpuSqlParser::SHOWTB);
    setState(148);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(146);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(147);
      database();
    }
    setState(150);
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
    setState(152);
    match(GpuSqlParser::SHOWCL);
    setState(153);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(154);
    table();
    setState(157);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
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
      database();
    }
    setState(159);
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
    setState(161);
    match(GpuSqlParser::SELECT);
    setState(162);
    selectColumns();
    setState(163);
    match(GpuSqlParser::FROM);
    setState(164);
    fromTables();
    setState(166);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::JOIN) {
      setState(165);
      joinClauses();
    }
    setState(170);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::WHERE) {
      setState(168);
      match(GpuSqlParser::WHERE);
      setState(169);
      whereClause();
    }
    setState(174);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::GROUPBY) {
      setState(172);
      match(GpuSqlParser::GROUPBY);
      setState(173);
      groupByColumns();
    }
    setState(178);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::ORDERBY) {
      setState(176);
      match(GpuSqlParser::ORDERBY);
      setState(177);
      orderByColumns();
    }
    setState(182);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::LIMIT) {
      setState(180);
      match(GpuSqlParser::LIMIT);
      setState(181);
      limit();
    }
    setState(186);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::OFFSET) {
      setState(184);
      match(GpuSqlParser::OFFSET);
      setState(185);
      offset();
    }
    setState(188);
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
    setState(190);
    match(GpuSqlParser::CREATEDB);
    setState(191);
    database();
    setState(192);
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
    setState(194);
    match(GpuSqlParser::DROPDB);
    setState(195);
    database();
    setState(196);
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
    setState(198);
    match(GpuSqlParser::CREATETABLE);
    setState(199);
    table();
    setState(200);
    match(GpuSqlParser::LPAREN);
    setState(201);
    newTableEntries();
    setState(202);
    match(GpuSqlParser::RPAREN);
    setState(203);
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
    setState(205);
    match(GpuSqlParser::DROPTABLE);
    setState(206);
    table();
    setState(207);
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
    setState(209);
    match(GpuSqlParser::ALTERTABLE);
    setState(210);
    table();
    setState(211);
    alterTableEntries();
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
    setState(214);
    match(GpuSqlParser::CREATEINDEX);
    setState(215);
    indexName();
    setState(216);
    match(GpuSqlParser::ON);
    setState(217);
    table();
    setState(218);
    match(GpuSqlParser::LPAREN);
    setState(219);
    indexColumns();
    setState(220);
    match(GpuSqlParser::RPAREN);
    setState(221);
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
    setState(223);
    match(GpuSqlParser::INSERTINTO);
    setState(224);
    table();
    setState(225);
    match(GpuSqlParser::LPAREN);
    setState(226);
    insertIntoColumns();
    setState(227);
    match(GpuSqlParser::RPAREN);
    setState(228);
    match(GpuSqlParser::VALUES);
    setState(229);
    match(GpuSqlParser::LPAREN);
    setState(230);
    insertIntoValues();
    setState(231);
    match(GpuSqlParser::RPAREN);
    setState(232);
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
    setState(234);
    newTableEntry();
    setState(239);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(235);
      match(GpuSqlParser::COMMA);
      setState(236);
      newTableEntry();
      setState(241);
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
    setState(244);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::ID: {
        setState(242);
        newTableColumn();
        break;
      }

      case GpuSqlParser::INDEX: {
        setState(243);
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
    setState(246);
    alterTableEntry();
    setState(251);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(247);
      match(GpuSqlParser::COMMA);
      setState(248);
      alterTableEntry();
      setState(253);
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
    setState(257);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::ADD: {
        setState(254);
        addColumn();
        break;
      }

      case GpuSqlParser::DROPCOLUMN: {
        setState(255);
        dropColumn();
        break;
      }

      case GpuSqlParser::ALTERCOLUMN: {
        setState(256);
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

GpuSqlParser::ColumnIdContext* GpuSqlParser::AddColumnContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
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
    setState(259);
    match(GpuSqlParser::ADD);
    setState(260);
    columnId();
    setState(261);
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

GpuSqlParser::ColumnIdContext* GpuSqlParser::DropColumnContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
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
    setState(263);
    match(GpuSqlParser::DROPCOLUMN);
    setState(264);
    columnId();
   
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

GpuSqlParser::ColumnIdContext* GpuSqlParser::AlterColumnContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
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
    setState(266);
    match(GpuSqlParser::ALTERCOLUMN);
    setState(267);
    columnId();
    setState(268);
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
  enterRule(_localctx, 42, GpuSqlParser::RuleNewTableColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(270);
    columnId();
    setState(271);
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
    setState(273);
    match(GpuSqlParser::INDEX);
    setState(274);
    indexName();
    setState(275);
    match(GpuSqlParser::LPAREN);
    setState(276);
    indexColumns();
    setState(277);
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
    setState(279);
    selectColumn();
    setState(284);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(280);
      match(GpuSqlParser::COMMA);
      setState(281);
      selectColumn();
      setState(286);
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
    setState(287);
    expression(0);
    setState(290);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(288);
      match(GpuSqlParser::AS);
      setState(289);
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
    setState(292);
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
    setState(294);
    orderByColumn();
    setState(299);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(295);
      match(GpuSqlParser::COMMA);
      setState(296);
      orderByColumn();
      setState(301);
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
    setState(302);
    columnId();
    setState(304);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::DIR) {
      setState(303);
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
    setState(306);
    columnValue();
    setState(311);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(307);
      match(GpuSqlParser::COMMA);
      setState(308);
      columnValue();
      setState(313);
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
    setState(314);
    columnId();
    setState(319);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(315);
      match(GpuSqlParser::COMMA);
      setState(316);
      columnId();
      setState(321);
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
    setState(322);
    column();
    setState(327);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(323);
      match(GpuSqlParser::COMMA);
      setState(324);
      column();
      setState(329);
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
    setState(330);
    groupByColumn();
    setState(335);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(331);
      match(GpuSqlParser::COMMA);
      setState(332);
      groupByColumn();
      setState(337);
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
    setState(338);
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
    setState(345);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(340);
      column();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(341);
      table();
      setState(342);
      match(GpuSqlParser::DOT);
      setState(343);
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
    setState(347);
    fromTable();
    setState(352);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(348);
      match(GpuSqlParser::COMMA);
      setState(349);
      fromTable();
      setState(354);
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
    setState(356); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(355);
      joinClause();
      setState(358); 
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
    setState(360);
    match(GpuSqlParser::JOIN);
    setState(361);
    joinTable();
    setState(362);
    match(GpuSqlParser::ON);
    setState(363);
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
    setState(365);
    table();
    setState(368);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(366);
      match(GpuSqlParser::AS);
      setState(367);
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
  enterRule(_localctx, 78, GpuSqlParser::RuleTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(375);
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
  enterRule(_localctx, 80, GpuSqlParser::RuleColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(377);
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
  enterRule(_localctx, 82, GpuSqlParser::RuleDatabase);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(379);
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
  enterRule(_localctx, 84, GpuSqlParser::RuleAlias);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(381);
    match(GpuSqlParser::ID);
   
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

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(383);
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
  enterRule(_localctx, 88, GpuSqlParser::RuleLimit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(385);
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
    setState(387);
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
  enterRule(_localctx, 92, GpuSqlParser::RuleColumnValue);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(393);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::INTLIT: {
        setState(389);
        match(GpuSqlParser::INTLIT);
        break;
      }

      case GpuSqlParser::FLOATLIT: {
        setState(390);
        match(GpuSqlParser::FLOATLIT);
        break;
      }

      case GpuSqlParser::POINT:
      case GpuSqlParser::MULTIPOINT:
      case GpuSqlParser::LINESTRING:
      case GpuSqlParser::MULTILINESTRING:
      case GpuSqlParser::POLYGON:
      case GpuSqlParser::MULTIPOLYGON: {
        setState(391);
        geometry();
        break;
      }

      case GpuSqlParser::STRINGLIT: {
        setState(392);
        match(GpuSqlParser::STRINGLIT);
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
  size_t startState = 94;
  enterRecursionRule(_localctx, 94, GpuSqlParser::RuleExpression, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(589);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 29, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(396);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOT);
      setState(397);
      expression(57);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(398);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUS);
      setState(399);
      expression(56);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(400);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ABS);
      setState(401);
      match(GpuSqlParser::LPAREN);
      setState(402);
      expression(0);
      setState(403);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(405);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIN);
      setState(406);
      match(GpuSqlParser::LPAREN);
      setState(407);
      expression(0);
      setState(408);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(410);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COS);
      setState(411);
      match(GpuSqlParser::LPAREN);
      setState(412);
      expression(0);
      setState(413);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(415);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::TAN);
      setState(416);
      match(GpuSqlParser::LPAREN);
      setState(417);
      expression(0);
      setState(418);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(420);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COT);
      setState(421);
      match(GpuSqlParser::LPAREN);
      setState(422);
      expression(0);
      setState(423);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(425);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ASIN);
      setState(426);
      match(GpuSqlParser::LPAREN);
      setState(427);
      expression(0);
      setState(428);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(430);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ACOS);
      setState(431);
      match(GpuSqlParser::LPAREN);
      setState(432);
      expression(0);
      setState(433);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(435);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN);
      setState(436);
      match(GpuSqlParser::LPAREN);
      setState(437);
      expression(0);
      setState(438);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 11: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(440);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG10);
      setState(441);
      match(GpuSqlParser::LPAREN);
      setState(442);
      expression(0);
      setState(443);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 12: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(445);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(446);
      match(GpuSqlParser::LPAREN);
      setState(447);
      expression(0);
      setState(448);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 13: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(450);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::EXP);
      setState(451);
      match(GpuSqlParser::LPAREN);
      setState(452);
      expression(0);
      setState(453);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 14: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(455);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQRT);
      setState(456);
      match(GpuSqlParser::LPAREN);
      setState(457);
      expression(0);
      setState(458);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 15: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(460);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQUARE);
      setState(461);
      match(GpuSqlParser::LPAREN);
      setState(462);
      expression(0);
      setState(463);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 16: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(465);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIGN);
      setState(466);
      match(GpuSqlParser::LPAREN);
      setState(467);
      expression(0);
      setState(468);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 17: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(470);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROUND);
      setState(471);
      match(GpuSqlParser::LPAREN);
      setState(472);
      expression(0);
      setState(473);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 18: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(475);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::FLOOR);
      setState(476);
      match(GpuSqlParser::LPAREN);
      setState(477);
      expression(0);
      setState(478);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 19: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(480);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::CEIL);
      setState(481);
      match(GpuSqlParser::LPAREN);
      setState(482);
      expression(0);
      setState(483);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 20: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(485);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::YEAR);
      setState(486);
      match(GpuSqlParser::LPAREN);
      setState(487);
      expression(0);
      setState(488);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 21: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(490);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MONTH);
      setState(491);
      match(GpuSqlParser::LPAREN);
      setState(492);
      expression(0);
      setState(493);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 22: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(495);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DAY);
      setState(496);
      match(GpuSqlParser::LPAREN);
      setState(497);
      expression(0);
      setState(498);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 23: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(500);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::HOUR);
      setState(501);
      match(GpuSqlParser::LPAREN);
      setState(502);
      expression(0);
      setState(503);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 24: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(505);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUTE);
      setState(506);
      match(GpuSqlParser::LPAREN);
      setState(507);
      expression(0);
      setState(508);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 25: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(510);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SECOND);
      setState(511);
      match(GpuSqlParser::LPAREN);
      setState(512);
      expression(0);
      setState(513);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 26: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(515);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN2);
      setState(516);
      match(GpuSqlParser::LPAREN);
      setState(517);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(518);
      match(GpuSqlParser::COMMA);
      setState(519);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(520);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 27: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(522);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(523);
      match(GpuSqlParser::LPAREN);
      setState(524);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(525);
      match(GpuSqlParser::COMMA);
      setState(526);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(527);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 28: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(529);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POW);
      setState(530);
      match(GpuSqlParser::LPAREN);
      setState(531);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(532);
      match(GpuSqlParser::COMMA);
      setState(533);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(534);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 29: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(536);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROOT);
      setState(537);
      match(GpuSqlParser::LPAREN);
      setState(538);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(539);
      match(GpuSqlParser::COMMA);
      setState(540);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(541);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 30: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(543);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POINT);
      setState(544);
      match(GpuSqlParser::LPAREN);
      setState(545);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(546);
      match(GpuSqlParser::COMMA);
      setState(547);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(548);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 31: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(550);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_CONTAINS);
      setState(551);
      match(GpuSqlParser::LPAREN);
      setState(552);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(553);
      match(GpuSqlParser::COMMA);
      setState(554);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(555);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 32: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(557);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_INTERSECT);
      setState(558);
      match(GpuSqlParser::LPAREN);
      setState(559);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(560);
      match(GpuSqlParser::COMMA);
      setState(561);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(562);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 33: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(564);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_UNION);
      setState(565);
      match(GpuSqlParser::LPAREN);
      setState(566);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(567);
      match(GpuSqlParser::COMMA);
      setState(568);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(569);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 34: {
      _localctx = _tracker.createInstance<ParenExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(571);
      match(GpuSqlParser::LPAREN);
      setState(572);
      expression(0);
      setState(573);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 35: {
      _localctx = _tracker.createInstance<VarReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(575);
      columnId();
      break;
    }

    case 36: {
      _localctx = _tracker.createInstance<GeoReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(576);
      geometry();
      break;
    }

    case 37: {
      _localctx = _tracker.createInstance<DateTimeLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(577);
      match(GpuSqlParser::DATETIMELIT);
      break;
    }

    case 38: {
      _localctx = _tracker.createInstance<DecimalLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(578);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 39: {
      _localctx = _tracker.createInstance<PiLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(579);
      match(GpuSqlParser::PI);
      break;
    }

    case 40: {
      _localctx = _tracker.createInstance<NowLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(580);
      match(GpuSqlParser::NOW);
      break;
    }

    case 41: {
      _localctx = _tracker.createInstance<IntLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(581);
      match(GpuSqlParser::INTLIT);
      break;
    }

    case 42: {
      _localctx = _tracker.createInstance<StringLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(582);
      match(GpuSqlParser::STRINGLIT);
      break;
    }

    case 43: {
      _localctx = _tracker.createInstance<BooleanLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(583);
      match(GpuSqlParser::BOOLEANLIT);
      break;
    }

    case 44: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(584);
      match(GpuSqlParser::AGG);
      setState(585);
      match(GpuSqlParser::LPAREN);
      setState(586);
      expression(0);
      setState(587);
      match(GpuSqlParser::RPAREN);
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(635);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(633);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(591);

          if (!(precpred(_ctx, 32))) throw FailedPredicateException(this, "precpred(_ctx, 32)");
          setState(592);
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
          setState(593);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(33);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(594);

          if (!(precpred(_ctx, 31))) throw FailedPredicateException(this, "precpred(_ctx, 31)");
          setState(595);
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
          setState(596);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(32);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(597);

          if (!(precpred(_ctx, 30))) throw FailedPredicateException(this, "precpred(_ctx, 30)");
          setState(598);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MODULO);
          setState(599);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(31);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(600);

          if (!(precpred(_ctx, 25))) throw FailedPredicateException(this, "precpred(_ctx, 25)");
          setState(601);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::XOR);
          setState(602);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(26);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(603);

          if (!(precpred(_ctx, 24))) throw FailedPredicateException(this, "precpred(_ctx, 24)");
          setState(604);
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
          setState(605);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(25);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(606);

          if (!(precpred(_ctx, 23))) throw FailedPredicateException(this, "precpred(_ctx, 23)");
          setState(607);
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
          setState(608);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(24);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(609);

          if (!(precpred(_ctx, 22))) throw FailedPredicateException(this, "precpred(_ctx, 22)");
          setState(610);
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
          setState(611);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(23);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(612);

          if (!(precpred(_ctx, 21))) throw FailedPredicateException(this, "precpred(_ctx, 21)");
          setState(613);
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
          setState(614);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(22);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(615);

          if (!(precpred(_ctx, 20))) throw FailedPredicateException(this, "precpred(_ctx, 20)");
          setState(616);
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
          setState(617);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(21);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(618);

          if (!(precpred(_ctx, 19))) throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(619);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOTEQUALS_GT_LT);
          setState(620);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(20);
          break;
        }

        case 11: {
          auto newContext = _tracker.createInstance<TernaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(621);

          if (!(precpred(_ctx, 14))) throw FailedPredicateException(this, "precpred(_ctx, 14)");
          setState(622);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op = match(GpuSqlParser::BETWEEN);
          setState(623);
          expression(0);
          setState(624);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op2 = match(GpuSqlParser::AND);
          setState(625);
          expression(15);
          break;
        }

        case 12: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(627);

          if (!(precpred(_ctx, 13))) throw FailedPredicateException(this, "precpred(_ctx, 13)");
          setState(628);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::AND);
          setState(629);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(14);
          break;
        }

        case 13: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(630);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(631);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::OR);
          setState(632);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(13);
          break;
        }

        } 
      }
      setState(637);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx);
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
  enterRule(_localctx, 96, GpuSqlParser::RuleGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(644);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::POINT: {
        setState(638);
        pointGeometry();
        break;
      }

      case GpuSqlParser::POLYGON: {
        setState(639);
        polygonGeometry();
        break;
      }

      case GpuSqlParser::LINESTRING: {
        setState(640);
        lineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOINT: {
        setState(641);
        multiPointGeometry();
        break;
      }

      case GpuSqlParser::MULTILINESTRING: {
        setState(642);
        multiLineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOLYGON: {
        setState(643);
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
  enterRule(_localctx, 98, GpuSqlParser::RulePointGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(646);
    match(GpuSqlParser::POINT);
    setState(647);
    match(GpuSqlParser::LPAREN);
    setState(648);
    point();
    setState(649);
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
  enterRule(_localctx, 100, GpuSqlParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(651);
    match(GpuSqlParser::LINESTRING);
    setState(652);
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
  enterRule(_localctx, 102, GpuSqlParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(654);
    match(GpuSqlParser::POLYGON);
    setState(655);
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
  enterRule(_localctx, 104, GpuSqlParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(657);
    match(GpuSqlParser::MULTIPOINT);
    setState(658);
    match(GpuSqlParser::LPAREN);
    setState(659);
    pointOrClosedPoint();
    setState(664);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(660);
      match(GpuSqlParser::COMMA);
      setState(661);
      pointOrClosedPoint();
      setState(666);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(667);
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
  enterRule(_localctx, 106, GpuSqlParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(669);
    match(GpuSqlParser::MULTILINESTRING);
    setState(670);
    match(GpuSqlParser::LPAREN);
    setState(671);
    lineString();
    setState(676);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(672);
      match(GpuSqlParser::COMMA);
      setState(673);
      lineString();
      setState(678);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(679);
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
  enterRule(_localctx, 108, GpuSqlParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(681);
    match(GpuSqlParser::MULTIPOLYGON);
    setState(682);
    match(GpuSqlParser::LPAREN);
    setState(683);
    polygon();
    setState(688);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(684);
      match(GpuSqlParser::COMMA);
      setState(685);
      polygon();
      setState(690);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(691);
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
  enterRule(_localctx, 110, GpuSqlParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(698);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT: {
        enterOuterAlt(_localctx, 1);
        setState(693);
        point();
        break;
      }

      case GpuSqlParser::LPAREN: {
        enterOuterAlt(_localctx, 2);
        setState(694);
        match(GpuSqlParser::LPAREN);
        setState(695);
        point();
        setState(696);
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
  enterRule(_localctx, 112, GpuSqlParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(700);
    match(GpuSqlParser::LPAREN);
    setState(701);
    lineString();
    setState(706);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(702);
      match(GpuSqlParser::COMMA);
      setState(703);
      lineString();
      setState(708);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(709);
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
  enterRule(_localctx, 114, GpuSqlParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(711);
    match(GpuSqlParser::LPAREN);
    setState(712);
    point();
    setState(717);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(713);
      match(GpuSqlParser::COMMA);
      setState(714);
      point();
      setState(719);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(720);
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
  enterRule(_localctx, 116, GpuSqlParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(722);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FLOATLIT

    || _la == GpuSqlParser::INTLIT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(723);
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
    case 47: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool GpuSqlParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 32);
    case 1: return precpred(_ctx, 31);
    case 2: return precpred(_ctx, 30);
    case 3: return precpred(_ctx, 25);
    case 4: return precpred(_ctx, 24);
    case 5: return precpred(_ctx, 23);
    case 6: return precpred(_ctx, 22);
    case 7: return precpred(_ctx, 21);
    case 8: return precpred(_ctx, 20);
    case 9: return precpred(_ctx, 19);
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
  "columnValue", "expression", "geometry", "pointGeometry", "lineStringGeometry", 
  "polygonGeometry", "multiPointGeometry", "multiLineStringGeometry", "multiPolygonGeometry", 
  "pointOrClosedPoint", "polygon", "lineString", "point"
};

std::vector<std::string> GpuSqlParser::_literalNames = {
  "", "", "'\n'", "'\r'", "'\r\n'", "", "';'", "'''", "'\"'", "'_'", "':'", 
  "','", "'.'", "", "", "'POINT'", "'MULTIPOINT'", "'LINESTRING'", "'MULTILINESTRING'", 
  "'POLYGON'", "'MULTIPOLYGON'", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "'+'", "'-'", "'*'", "'/'", 
  "'%'", "'^'", "'='", "'!='", "'<>'", "'('", "')'", "'>'", "'<'", "'>='", 
  "'<='", "'!'", "", "", "'|'", "'&'", "'<<'", "'>>'"
};

std::vector<std::string> GpuSqlParser::_symbolicNames = {
  "", "DATETIMELIT", "LF", "CR", "CRLF", "WS", "SEMICOL", "SQOUTE", "DQOUTE", 
  "UNDERSCORE", "COLON", "COMMA", "DOT", "DATELIT", "TIMELIT", "POINT", 
  "MULTIPOINT", "LINESTRING", "MULTILINESTRING", "POLYGON", "MULTIPOLYGON", 
  "DATATYPE", "INTTYPE", "LONGTYPE", "FLOATTYPE", "DOUBLETYPE", "STRINGTYPE", 
  "BOOLEANTYPE", "POINTTYPE", "POLYTYPE", "INSERTINTO", "CREATEDB", "DROPDB", 
  "CREATETABLE", "DROPTABLE", "ALTERTABLE", "ADD", "DROPCOLUMN", "ALTERCOLUMN", 
  "CREATEINDEX", "INDEX", "PRIMARYKEY", "VALUES", "SELECT", "FROM", "JOIN", 
  "WHERE", "GROUPBY", "AS", "IN", "BETWEEN", "ON", "ORDERBY", "DIR", "LIMIT", 
  "OFFSET", "SHOWDB", "SHOWTB", "SHOWCL", "AGG", "AVG", "SUM", "MIN", "MAX", 
  "COUNT", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "NOW", "PI", 
  "ABS", "SIN", "COS", "TAN", "COT", "ASIN", "ACOS", "ATAN", "ATAN2", "LOG10", 
  "LOG", "EXP", "POW", "SQRT", "SQUARE", "SIGN", "ROOT", "ROUND", "CEIL", 
  "FLOOR", "GEO_CONTAINS", "GEO_INTERSECT", "GEO_UNION", "PLUS", "MINUS", 
  "ASTERISK", "DIVISION", "MODULO", "XOR", "EQUALS", "NOTEQUALS", "NOTEQUALS_GT_LT", 
  "LPAREN", "RPAREN", "GREATER", "LESS", "GREATEREQ", "LESSEQ", "NOT", "OR", 
  "AND", "BIT_OR", "BIT_AND", "L_SHIFT", "R_SHIFT", "FLOATLIT", "INTLIT", 
  "ID", "BOOLEANLIT", "STRINGLIT"
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
    0x3, 0x7c, 0x2d8, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x9, 0x3c, 0x3, 0x2, 0x7, 0x2, 0x7a, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x7d, 
    0xb, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x8a, 0xa, 
    0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 0x8f, 0xa, 0x4, 0x3, 0x5, 
    0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x97, 0xa, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x7, 0x5, 0x7, 0xa0, 0xa, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xa9, 0xa, 0x8, 0x3, 0x8, 0x3, 
    0x8, 0x5, 0x8, 0xad, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb1, 0xa, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb5, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x5, 0x8, 0xb9, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xbd, 0xa, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 
    0x7, 0x10, 0xf0, 0xa, 0x10, 0xc, 0x10, 0xe, 0x10, 0xf3, 0xb, 0x10, 0x3, 
    0x11, 0x3, 0x11, 0x5, 0x11, 0xf7, 0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 
    0x12, 0x7, 0x12, 0xfc, 0xa, 0x12, 0xc, 0x12, 0xe, 0x12, 0xff, 0xb, 0x12, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x5, 0x13, 0x104, 0xa, 0x13, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 
    0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 
    0x19, 0x3, 0x19, 0x3, 0x19, 0x7, 0x19, 0x11d, 0xa, 0x19, 0xc, 0x19, 
    0xe, 0x19, 0x120, 0xb, 0x19, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 
    0x125, 0xa, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
    0x7, 0x1c, 0x12c, 0xa, 0x1c, 0xc, 0x1c, 0xe, 0x1c, 0x12f, 0xb, 0x1c, 
    0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x133, 0xa, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x7, 0x1e, 0x138, 0xa, 0x1e, 0xc, 0x1e, 0xe, 0x1e, 0x13b, 
    0xb, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x7, 0x1f, 0x140, 0xa, 0x1f, 
    0xc, 0x1f, 0xe, 0x1f, 0x143, 0xb, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
    0x7, 0x20, 0x148, 0xa, 0x20, 0xc, 0x20, 0xe, 0x20, 0x14b, 0xb, 0x20, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x7, 0x21, 0x150, 0xa, 0x21, 0xc, 0x21, 
    0xe, 0x21, 0x153, 0xb, 0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x5, 0x23, 0x15c, 0xa, 0x23, 0x3, 0x24, 
    0x3, 0x24, 0x3, 0x24, 0x7, 0x24, 0x161, 0xa, 0x24, 0xc, 0x24, 0xe, 0x24, 
    0x164, 0xb, 0x24, 0x3, 0x25, 0x6, 0x25, 0x167, 0xa, 0x25, 0xd, 0x25, 
    0xe, 0x25, 0x168, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 
    0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x5, 0x27, 0x173, 0xa, 0x27, 0x3, 0x28, 
    0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 0x178, 0xa, 0x28, 0x3, 0x29, 0x3, 0x29, 
    0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 
    0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x30, 
    0x3, 0x30, 0x3, 0x30, 0x3, 0x30, 0x5, 0x30, 0x18c, 0xa, 0x30, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x5, 0x31, 0x250, 
    0xa, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 
    0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x7, 0x31, 0x27c, 0xa, 0x31, 
    0xc, 0x31, 0xe, 0x31, 0x27f, 0xb, 0x31, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x5, 0x32, 0x287, 0xa, 0x32, 0x3, 0x33, 
    0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x34, 0x3, 0x34, 0x3, 
    0x34, 0x3, 0x35, 0x3, 0x35, 0x3, 0x35, 0x3, 0x36, 0x3, 0x36, 0x3, 0x36, 
    0x3, 0x36, 0x3, 0x36, 0x7, 0x36, 0x299, 0xa, 0x36, 0xc, 0x36, 0xe, 0x36, 
    0x29c, 0xb, 0x36, 0x3, 0x36, 0x3, 0x36, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x7, 0x37, 0x2a5, 0xa, 0x37, 0xc, 0x37, 0xe, 0x37, 
    0x2a8, 0xb, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 
    0x3, 0x38, 0x3, 0x38, 0x7, 0x38, 0x2b1, 0xa, 0x38, 0xc, 0x38, 0xe, 0x38, 
    0x2b4, 0xb, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x39, 0x3, 0x39, 0x3, 0x39, 
    0x3, 0x39, 0x3, 0x39, 0x5, 0x39, 0x2bd, 0xa, 0x39, 0x3, 0x3a, 0x3, 0x3a, 
    0x3, 0x3a, 0x3, 0x3a, 0x7, 0x3a, 0x2c3, 0xa, 0x3a, 0xc, 0x3a, 0xe, 0x3a, 
    0x2c6, 0xb, 0x3a, 0x3, 0x3a, 0x3, 0x3a, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 
    0x3, 0x3b, 0x7, 0x3b, 0x2ce, 0xa, 0x3b, 0xc, 0x3b, 0xe, 0x3b, 0x2d1, 
    0xb, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x3, 
    0x3c, 0x2, 0x3, 0x60, 0x3d, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 
    0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 
    0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 
    0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 
    0x5a, 0x5c, 0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 
    0x72, 0x74, 0x76, 0x2, 0xb, 0x4, 0x2, 0x2e, 0x2e, 0x33, 0x33, 0x3, 0x2, 
    0x64, 0x65, 0x3, 0x2, 0x62, 0x63, 0x3, 0x2, 0x74, 0x75, 0x3, 0x2, 0x76, 
    0x77, 0x3, 0x2, 0x6d, 0x6e, 0x3, 0x2, 0x6f, 0x70, 0x3, 0x2, 0x68, 0x69, 
    0x3, 0x2, 0x78, 0x79, 0x2, 0x307, 0x2, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x4, 
    0x89, 0x3, 0x2, 0x2, 0x2, 0x6, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x8, 0x90, 
    0x3, 0x2, 0x2, 0x2, 0xa, 0x93, 0x3, 0x2, 0x2, 0x2, 0xc, 0x9a, 0x3, 0x2, 
    0x2, 0x2, 0xe, 0xa3, 0x3, 0x2, 0x2, 0x2, 0x10, 0xc0, 0x3, 0x2, 0x2, 
    0x2, 0x12, 0xc4, 0x3, 0x2, 0x2, 0x2, 0x14, 0xc8, 0x3, 0x2, 0x2, 0x2, 
    0x16, 0xcf, 0x3, 0x2, 0x2, 0x2, 0x18, 0xd3, 0x3, 0x2, 0x2, 0x2, 0x1a, 
    0xd8, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xe1, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xec, 
    0x3, 0x2, 0x2, 0x2, 0x20, 0xf6, 0x3, 0x2, 0x2, 0x2, 0x22, 0xf8, 0x3, 
    0x2, 0x2, 0x2, 0x24, 0x103, 0x3, 0x2, 0x2, 0x2, 0x26, 0x105, 0x3, 0x2, 
    0x2, 0x2, 0x28, 0x109, 0x3, 0x2, 0x2, 0x2, 0x2a, 0x10c, 0x3, 0x2, 0x2, 
    0x2, 0x2c, 0x110, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x113, 0x3, 0x2, 0x2, 0x2, 
    0x30, 0x119, 0x3, 0x2, 0x2, 0x2, 0x32, 0x121, 0x3, 0x2, 0x2, 0x2, 0x34, 
    0x126, 0x3, 0x2, 0x2, 0x2, 0x36, 0x128, 0x3, 0x2, 0x2, 0x2, 0x38, 0x130, 
    0x3, 0x2, 0x2, 0x2, 0x3a, 0x134, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x13c, 0x3, 
    0x2, 0x2, 0x2, 0x3e, 0x144, 0x3, 0x2, 0x2, 0x2, 0x40, 0x14c, 0x3, 0x2, 
    0x2, 0x2, 0x42, 0x154, 0x3, 0x2, 0x2, 0x2, 0x44, 0x15b, 0x3, 0x2, 0x2, 
    0x2, 0x46, 0x15d, 0x3, 0x2, 0x2, 0x2, 0x48, 0x166, 0x3, 0x2, 0x2, 0x2, 
    0x4a, 0x16a, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x16f, 0x3, 0x2, 0x2, 0x2, 0x4e, 
    0x174, 0x3, 0x2, 0x2, 0x2, 0x50, 0x179, 0x3, 0x2, 0x2, 0x2, 0x52, 0x17b, 
    0x3, 0x2, 0x2, 0x2, 0x54, 0x17d, 0x3, 0x2, 0x2, 0x2, 0x56, 0x17f, 0x3, 
    0x2, 0x2, 0x2, 0x58, 0x181, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x183, 0x3, 0x2, 
    0x2, 0x2, 0x5c, 0x185, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x18b, 0x3, 0x2, 0x2, 
    0x2, 0x60, 0x24f, 0x3, 0x2, 0x2, 0x2, 0x62, 0x286, 0x3, 0x2, 0x2, 0x2, 
    0x64, 0x288, 0x3, 0x2, 0x2, 0x2, 0x66, 0x28d, 0x3, 0x2, 0x2, 0x2, 0x68, 
    0x290, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x293, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x29f, 
    0x3, 0x2, 0x2, 0x2, 0x6e, 0x2ab, 0x3, 0x2, 0x2, 0x2, 0x70, 0x2bc, 0x3, 
    0x2, 0x2, 0x2, 0x72, 0x2be, 0x3, 0x2, 0x2, 0x2, 0x74, 0x2c9, 0x3, 0x2, 
    0x2, 0x2, 0x76, 0x2d4, 0x3, 0x2, 0x2, 0x2, 0x78, 0x7a, 0x5, 0x4, 0x3, 
    0x2, 0x79, 0x78, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x7d, 0x3, 0x2, 0x2, 0x2, 
    0x7b, 0x79, 0x3, 0x2, 0x2, 0x2, 0x7b, 0x7c, 0x3, 0x2, 0x2, 0x2, 0x7c, 
    0x7e, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x7f, 
    0x7, 0x2, 0x2, 0x3, 0x7f, 0x3, 0x3, 0x2, 0x2, 0x2, 0x80, 0x8a, 0x5, 
    0xe, 0x8, 0x2, 0x81, 0x8a, 0x5, 0x10, 0x9, 0x2, 0x82, 0x8a, 0x5, 0x12, 
    0xa, 0x2, 0x83, 0x8a, 0x5, 0x14, 0xb, 0x2, 0x84, 0x8a, 0x5, 0x16, 0xc, 
    0x2, 0x85, 0x8a, 0x5, 0x18, 0xd, 0x2, 0x86, 0x8a, 0x5, 0x1a, 0xe, 0x2, 
    0x87, 0x8a, 0x5, 0x1c, 0xf, 0x2, 0x88, 0x8a, 0x5, 0x6, 0x4, 0x2, 0x89, 
    0x80, 0x3, 0x2, 0x2, 0x2, 0x89, 0x81, 0x3, 0x2, 0x2, 0x2, 0x89, 0x82, 
    0x3, 0x2, 0x2, 0x2, 0x89, 0x83, 0x3, 0x2, 0x2, 0x2, 0x89, 0x84, 0x3, 
    0x2, 0x2, 0x2, 0x89, 0x85, 0x3, 0x2, 0x2, 0x2, 0x89, 0x86, 0x3, 0x2, 
    0x2, 0x2, 0x89, 0x87, 0x3, 0x2, 0x2, 0x2, 0x89, 0x88, 0x3, 0x2, 0x2, 
    0x2, 0x8a, 0x5, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x8f, 0x5, 0x8, 0x5, 0x2, 
    0x8c, 0x8f, 0x5, 0xa, 0x6, 0x2, 0x8d, 0x8f, 0x5, 0xc, 0x7, 0x2, 0x8e, 
    0x8b, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x8c, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x8d, 
    0x3, 0x2, 0x2, 0x2, 0x8f, 0x7, 0x3, 0x2, 0x2, 0x2, 0x90, 0x91, 0x7, 
    0x3a, 0x2, 0x2, 0x91, 0x92, 0x7, 0x8, 0x2, 0x2, 0x92, 0x9, 0x3, 0x2, 
    0x2, 0x2, 0x93, 0x96, 0x7, 0x3b, 0x2, 0x2, 0x94, 0x95, 0x9, 0x2, 0x2, 
    0x2, 0x95, 0x97, 0x5, 0x54, 0x2b, 0x2, 0x96, 0x94, 0x3, 0x2, 0x2, 0x2, 
    0x96, 0x97, 0x3, 0x2, 0x2, 0x2, 0x97, 0x98, 0x3, 0x2, 0x2, 0x2, 0x98, 
    0x99, 0x7, 0x8, 0x2, 0x2, 0x99, 0xb, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x9b, 
    0x7, 0x3c, 0x2, 0x2, 0x9b, 0x9c, 0x9, 0x2, 0x2, 0x2, 0x9c, 0x9f, 0x5, 
    0x50, 0x29, 0x2, 0x9d, 0x9e, 0x9, 0x2, 0x2, 0x2, 0x9e, 0xa0, 0x5, 0x54, 
    0x2b, 0x2, 0x9f, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x9f, 0xa0, 0x3, 0x2, 0x2, 
    0x2, 0xa0, 0xa1, 0x3, 0x2, 0x2, 0x2, 0xa1, 0xa2, 0x7, 0x8, 0x2, 0x2, 
    0xa2, 0xd, 0x3, 0x2, 0x2, 0x2, 0xa3, 0xa4, 0x7, 0x2d, 0x2, 0x2, 0xa4, 
    0xa5, 0x5, 0x30, 0x19, 0x2, 0xa5, 0xa6, 0x7, 0x2e, 0x2, 0x2, 0xa6, 0xa8, 
    0x5, 0x46, 0x24, 0x2, 0xa7, 0xa9, 0x5, 0x48, 0x25, 0x2, 0xa8, 0xa7, 
    0x3, 0x2, 0x2, 0x2, 0xa8, 0xa9, 0x3, 0x2, 0x2, 0x2, 0xa9, 0xac, 0x3, 
    0x2, 0x2, 0x2, 0xaa, 0xab, 0x7, 0x30, 0x2, 0x2, 0xab, 0xad, 0x5, 0x34, 
    0x1b, 0x2, 0xac, 0xaa, 0x3, 0x2, 0x2, 0x2, 0xac, 0xad, 0x3, 0x2, 0x2, 
    0x2, 0xad, 0xb0, 0x3, 0x2, 0x2, 0x2, 0xae, 0xaf, 0x7, 0x31, 0x2, 0x2, 
    0xaf, 0xb1, 0x5, 0x40, 0x21, 0x2, 0xb0, 0xae, 0x3, 0x2, 0x2, 0x2, 0xb0, 
    0xb1, 0x3, 0x2, 0x2, 0x2, 0xb1, 0xb4, 0x3, 0x2, 0x2, 0x2, 0xb2, 0xb3, 
    0x7, 0x36, 0x2, 0x2, 0xb3, 0xb5, 0x5, 0x36, 0x1c, 0x2, 0xb4, 0xb2, 0x3, 
    0x2, 0x2, 0x2, 0xb4, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xb8, 0x3, 0x2, 
    0x2, 0x2, 0xb6, 0xb7, 0x7, 0x38, 0x2, 0x2, 0xb7, 0xb9, 0x5, 0x5a, 0x2e, 
    0x2, 0xb8, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xb9, 0x3, 0x2, 0x2, 0x2, 
    0xb9, 0xbc, 0x3, 0x2, 0x2, 0x2, 0xba, 0xbb, 0x7, 0x39, 0x2, 0x2, 0xbb, 
    0xbd, 0x5, 0x5c, 0x2f, 0x2, 0xbc, 0xba, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xbd, 
    0x3, 0x2, 0x2, 0x2, 0xbd, 0xbe, 0x3, 0x2, 0x2, 0x2, 0xbe, 0xbf, 0x7, 
    0x8, 0x2, 0x2, 0xbf, 0xf, 0x3, 0x2, 0x2, 0x2, 0xc0, 0xc1, 0x7, 0x21, 
    0x2, 0x2, 0xc1, 0xc2, 0x5, 0x54, 0x2b, 0x2, 0xc2, 0xc3, 0x7, 0x8, 0x2, 
    0x2, 0xc3, 0x11, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc5, 0x7, 0x22, 0x2, 0x2, 
    0xc5, 0xc6, 0x5, 0x54, 0x2b, 0x2, 0xc6, 0xc7, 0x7, 0x8, 0x2, 0x2, 0xc7, 
    0x13, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xc9, 0x7, 0x23, 0x2, 0x2, 0xc9, 0xca, 
    0x5, 0x50, 0x29, 0x2, 0xca, 0xcb, 0x7, 0x6b, 0x2, 0x2, 0xcb, 0xcc, 0x5, 
    0x1e, 0x10, 0x2, 0xcc, 0xcd, 0x7, 0x6c, 0x2, 0x2, 0xcd, 0xce, 0x7, 0x8, 
    0x2, 0x2, 0xce, 0x15, 0x3, 0x2, 0x2, 0x2, 0xcf, 0xd0, 0x7, 0x24, 0x2, 
    0x2, 0xd0, 0xd1, 0x5, 0x50, 0x29, 0x2, 0xd1, 0xd2, 0x7, 0x8, 0x2, 0x2, 
    0xd2, 0x17, 0x3, 0x2, 0x2, 0x2, 0xd3, 0xd4, 0x7, 0x25, 0x2, 0x2, 0xd4, 
    0xd5, 0x5, 0x50, 0x29, 0x2, 0xd5, 0xd6, 0x5, 0x22, 0x12, 0x2, 0xd6, 
    0xd7, 0x7, 0x8, 0x2, 0x2, 0xd7, 0x19, 0x3, 0x2, 0x2, 0x2, 0xd8, 0xd9, 
    0x7, 0x29, 0x2, 0x2, 0xd9, 0xda, 0x5, 0x58, 0x2d, 0x2, 0xda, 0xdb, 0x7, 
    0x35, 0x2, 0x2, 0xdb, 0xdc, 0x5, 0x50, 0x29, 0x2, 0xdc, 0xdd, 0x7, 0x6b, 
    0x2, 0x2, 0xdd, 0xde, 0x5, 0x3e, 0x20, 0x2, 0xde, 0xdf, 0x7, 0x6c, 0x2, 
    0x2, 0xdf, 0xe0, 0x7, 0x8, 0x2, 0x2, 0xe0, 0x1b, 0x3, 0x2, 0x2, 0x2, 
    0xe1, 0xe2, 0x7, 0x20, 0x2, 0x2, 0xe2, 0xe3, 0x5, 0x50, 0x29, 0x2, 0xe3, 
    0xe4, 0x7, 0x6b, 0x2, 0x2, 0xe4, 0xe5, 0x5, 0x3c, 0x1f, 0x2, 0xe5, 0xe6, 
    0x7, 0x6c, 0x2, 0x2, 0xe6, 0xe7, 0x7, 0x2c, 0x2, 0x2, 0xe7, 0xe8, 0x7, 
    0x6b, 0x2, 0x2, 0xe8, 0xe9, 0x5, 0x3a, 0x1e, 0x2, 0xe9, 0xea, 0x7, 0x6c, 
    0x2, 0x2, 0xea, 0xeb, 0x7, 0x8, 0x2, 0x2, 0xeb, 0x1d, 0x3, 0x2, 0x2, 
    0x2, 0xec, 0xf1, 0x5, 0x20, 0x11, 0x2, 0xed, 0xee, 0x7, 0xd, 0x2, 0x2, 
    0xee, 0xf0, 0x5, 0x20, 0x11, 0x2, 0xef, 0xed, 0x3, 0x2, 0x2, 0x2, 0xf0, 
    0xf3, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xef, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xf2, 
    0x3, 0x2, 0x2, 0x2, 0xf2, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xf3, 0xf1, 0x3, 
    0x2, 0x2, 0x2, 0xf4, 0xf7, 0x5, 0x2c, 0x17, 0x2, 0xf5, 0xf7, 0x5, 0x2e, 
    0x18, 0x2, 0xf6, 0xf4, 0x3, 0x2, 0x2, 0x2, 0xf6, 0xf5, 0x3, 0x2, 0x2, 
    0x2, 0xf7, 0x21, 0x3, 0x2, 0x2, 0x2, 0xf8, 0xfd, 0x5, 0x24, 0x13, 0x2, 
    0xf9, 0xfa, 0x7, 0xd, 0x2, 0x2, 0xfa, 0xfc, 0x5, 0x24, 0x13, 0x2, 0xfb, 
    0xf9, 0x3, 0x2, 0x2, 0x2, 0xfc, 0xff, 0x3, 0x2, 0x2, 0x2, 0xfd, 0xfb, 
    0x3, 0x2, 0x2, 0x2, 0xfd, 0xfe, 0x3, 0x2, 0x2, 0x2, 0xfe, 0x23, 0x3, 
    0x2, 0x2, 0x2, 0xff, 0xfd, 0x3, 0x2, 0x2, 0x2, 0x100, 0x104, 0x5, 0x26, 
    0x14, 0x2, 0x101, 0x104, 0x5, 0x28, 0x15, 0x2, 0x102, 0x104, 0x5, 0x2a, 
    0x16, 0x2, 0x103, 0x100, 0x3, 0x2, 0x2, 0x2, 0x103, 0x101, 0x3, 0x2, 
    0x2, 0x2, 0x103, 0x102, 0x3, 0x2, 0x2, 0x2, 0x104, 0x25, 0x3, 0x2, 0x2, 
    0x2, 0x105, 0x106, 0x7, 0x26, 0x2, 0x2, 0x106, 0x107, 0x5, 0x44, 0x23, 
    0x2, 0x107, 0x108, 0x7, 0x17, 0x2, 0x2, 0x108, 0x27, 0x3, 0x2, 0x2, 
    0x2, 0x109, 0x10a, 0x7, 0x27, 0x2, 0x2, 0x10a, 0x10b, 0x5, 0x44, 0x23, 
    0x2, 0x10b, 0x29, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10d, 0x7, 0x28, 0x2, 
    0x2, 0x10d, 0x10e, 0x5, 0x44, 0x23, 0x2, 0x10e, 0x10f, 0x7, 0x17, 0x2, 
    0x2, 0x10f, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x110, 0x111, 0x5, 0x44, 0x23, 
    0x2, 0x111, 0x112, 0x7, 0x17, 0x2, 0x2, 0x112, 0x2d, 0x3, 0x2, 0x2, 
    0x2, 0x113, 0x114, 0x7, 0x2a, 0x2, 0x2, 0x114, 0x115, 0x5, 0x58, 0x2d, 
    0x2, 0x115, 0x116, 0x7, 0x6b, 0x2, 0x2, 0x116, 0x117, 0x5, 0x3e, 0x20, 
    0x2, 0x117, 0x118, 0x7, 0x6c, 0x2, 0x2, 0x118, 0x2f, 0x3, 0x2, 0x2, 
    0x2, 0x119, 0x11e, 0x5, 0x32, 0x1a, 0x2, 0x11a, 0x11b, 0x7, 0xd, 0x2, 
    0x2, 0x11b, 0x11d, 0x5, 0x32, 0x1a, 0x2, 0x11c, 0x11a, 0x3, 0x2, 0x2, 
    0x2, 0x11d, 0x120, 0x3, 0x2, 0x2, 0x2, 0x11e, 0x11c, 0x3, 0x2, 0x2, 
    0x2, 0x11e, 0x11f, 0x3, 0x2, 0x2, 0x2, 0x11f, 0x31, 0x3, 0x2, 0x2, 0x2, 
    0x120, 0x11e, 0x3, 0x2, 0x2, 0x2, 0x121, 0x124, 0x5, 0x60, 0x31, 0x2, 
    0x122, 0x123, 0x7, 0x32, 0x2, 0x2, 0x123, 0x125, 0x5, 0x56, 0x2c, 0x2, 
    0x124, 0x122, 0x3, 0x2, 0x2, 0x2, 0x124, 0x125, 0x3, 0x2, 0x2, 0x2, 
    0x125, 0x33, 0x3, 0x2, 0x2, 0x2, 0x126, 0x127, 0x5, 0x60, 0x31, 0x2, 
    0x127, 0x35, 0x3, 0x2, 0x2, 0x2, 0x128, 0x12d, 0x5, 0x38, 0x1d, 0x2, 
    0x129, 0x12a, 0x7, 0xd, 0x2, 0x2, 0x12a, 0x12c, 0x5, 0x38, 0x1d, 0x2, 
    0x12b, 0x129, 0x3, 0x2, 0x2, 0x2, 0x12c, 0x12f, 0x3, 0x2, 0x2, 0x2, 
    0x12d, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x12e, 0x3, 0x2, 0x2, 0x2, 
    0x12e, 0x37, 0x3, 0x2, 0x2, 0x2, 0x12f, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x130, 
    0x132, 0x5, 0x44, 0x23, 0x2, 0x131, 0x133, 0x7, 0x37, 0x2, 0x2, 0x132, 
    0x131, 0x3, 0x2, 0x2, 0x2, 0x132, 0x133, 0x3, 0x2, 0x2, 0x2, 0x133, 
    0x39, 0x3, 0x2, 0x2, 0x2, 0x134, 0x139, 0x5, 0x5e, 0x30, 0x2, 0x135, 
    0x136, 0x7, 0xd, 0x2, 0x2, 0x136, 0x138, 0x5, 0x5e, 0x30, 0x2, 0x137, 
    0x135, 0x3, 0x2, 0x2, 0x2, 0x138, 0x13b, 0x3, 0x2, 0x2, 0x2, 0x139, 
    0x137, 0x3, 0x2, 0x2, 0x2, 0x139, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x13a, 
    0x3b, 0x3, 0x2, 0x2, 0x2, 0x13b, 0x139, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x141, 
    0x5, 0x44, 0x23, 0x2, 0x13d, 0x13e, 0x7, 0xd, 0x2, 0x2, 0x13e, 0x140, 
    0x5, 0x44, 0x23, 0x2, 0x13f, 0x13d, 0x3, 0x2, 0x2, 0x2, 0x140, 0x143, 
    0x3, 0x2, 0x2, 0x2, 0x141, 0x13f, 0x3, 0x2, 0x2, 0x2, 0x141, 0x142, 
    0x3, 0x2, 0x2, 0x2, 0x142, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x143, 0x141, 0x3, 
    0x2, 0x2, 0x2, 0x144, 0x149, 0x5, 0x52, 0x2a, 0x2, 0x145, 0x146, 0x7, 
    0xd, 0x2, 0x2, 0x146, 0x148, 0x5, 0x52, 0x2a, 0x2, 0x147, 0x145, 0x3, 
    0x2, 0x2, 0x2, 0x148, 0x14b, 0x3, 0x2, 0x2, 0x2, 0x149, 0x147, 0x3, 
    0x2, 0x2, 0x2, 0x149, 0x14a, 0x3, 0x2, 0x2, 0x2, 0x14a, 0x3f, 0x3, 0x2, 
    0x2, 0x2, 0x14b, 0x149, 0x3, 0x2, 0x2, 0x2, 0x14c, 0x151, 0x5, 0x42, 
    0x22, 0x2, 0x14d, 0x14e, 0x7, 0xd, 0x2, 0x2, 0x14e, 0x150, 0x5, 0x42, 
    0x22, 0x2, 0x14f, 0x14d, 0x3, 0x2, 0x2, 0x2, 0x150, 0x153, 0x3, 0x2, 
    0x2, 0x2, 0x151, 0x14f, 0x3, 0x2, 0x2, 0x2, 0x151, 0x152, 0x3, 0x2, 
    0x2, 0x2, 0x152, 0x41, 0x3, 0x2, 0x2, 0x2, 0x153, 0x151, 0x3, 0x2, 0x2, 
    0x2, 0x154, 0x155, 0x5, 0x60, 0x31, 0x2, 0x155, 0x43, 0x3, 0x2, 0x2, 
    0x2, 0x156, 0x15c, 0x5, 0x52, 0x2a, 0x2, 0x157, 0x158, 0x5, 0x50, 0x29, 
    0x2, 0x158, 0x159, 0x7, 0xe, 0x2, 0x2, 0x159, 0x15a, 0x5, 0x52, 0x2a, 
    0x2, 0x15a, 0x15c, 0x3, 0x2, 0x2, 0x2, 0x15b, 0x156, 0x3, 0x2, 0x2, 
    0x2, 0x15b, 0x157, 0x3, 0x2, 0x2, 0x2, 0x15c, 0x45, 0x3, 0x2, 0x2, 0x2, 
    0x15d, 0x162, 0x5, 0x4e, 0x28, 0x2, 0x15e, 0x15f, 0x7, 0xd, 0x2, 0x2, 
    0x15f, 0x161, 0x5, 0x4e, 0x28, 0x2, 0x160, 0x15e, 0x3, 0x2, 0x2, 0x2, 
    0x161, 0x164, 0x3, 0x2, 0x2, 0x2, 0x162, 0x160, 0x3, 0x2, 0x2, 0x2, 
    0x162, 0x163, 0x3, 0x2, 0x2, 0x2, 0x163, 0x47, 0x3, 0x2, 0x2, 0x2, 0x164, 
    0x162, 0x3, 0x2, 0x2, 0x2, 0x165, 0x167, 0x5, 0x4a, 0x26, 0x2, 0x166, 
    0x165, 0x3, 0x2, 0x2, 0x2, 0x167, 0x168, 0x3, 0x2, 0x2, 0x2, 0x168, 
    0x166, 0x3, 0x2, 0x2, 0x2, 0x168, 0x169, 0x3, 0x2, 0x2, 0x2, 0x169, 
    0x49, 0x3, 0x2, 0x2, 0x2, 0x16a, 0x16b, 0x7, 0x2f, 0x2, 0x2, 0x16b, 
    0x16c, 0x5, 0x4c, 0x27, 0x2, 0x16c, 0x16d, 0x7, 0x35, 0x2, 0x2, 0x16d, 
    0x16e, 0x5, 0x60, 0x31, 0x2, 0x16e, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x16f, 
    0x172, 0x5, 0x50, 0x29, 0x2, 0x170, 0x171, 0x7, 0x32, 0x2, 0x2, 0x171, 
    0x173, 0x5, 0x56, 0x2c, 0x2, 0x172, 0x170, 0x3, 0x2, 0x2, 0x2, 0x172, 
    0x173, 0x3, 0x2, 0x2, 0x2, 0x173, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x174, 0x177, 
    0x5, 0x50, 0x29, 0x2, 0x175, 0x176, 0x7, 0x32, 0x2, 0x2, 0x176, 0x178, 
    0x5, 0x56, 0x2c, 0x2, 0x177, 0x175, 0x3, 0x2, 0x2, 0x2, 0x177, 0x178, 
    0x3, 0x2, 0x2, 0x2, 0x178, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x179, 0x17a, 0x7, 
    0x7a, 0x2, 0x2, 0x17a, 0x51, 0x3, 0x2, 0x2, 0x2, 0x17b, 0x17c, 0x7, 
    0x7a, 0x2, 0x2, 0x17c, 0x53, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x17e, 0x7, 
    0x7a, 0x2, 0x2, 0x17e, 0x55, 0x3, 0x2, 0x2, 0x2, 0x17f, 0x180, 0x7, 
    0x7a, 0x2, 0x2, 0x180, 0x57, 0x3, 0x2, 0x2, 0x2, 0x181, 0x182, 0x7, 
    0x7a, 0x2, 0x2, 0x182, 0x59, 0x3, 0x2, 0x2, 0x2, 0x183, 0x184, 0x7, 
    0x79, 0x2, 0x2, 0x184, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x185, 0x186, 0x7, 
    0x79, 0x2, 0x2, 0x186, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x187, 0x18c, 0x7, 
    0x79, 0x2, 0x2, 0x188, 0x18c, 0x7, 0x78, 0x2, 0x2, 0x189, 0x18c, 0x5, 
    0x62, 0x32, 0x2, 0x18a, 0x18c, 0x7, 0x7c, 0x2, 0x2, 0x18b, 0x187, 0x3, 
    0x2, 0x2, 0x2, 0x18b, 0x188, 0x3, 0x2, 0x2, 0x2, 0x18b, 0x189, 0x3, 
    0x2, 0x2, 0x2, 0x18b, 0x18a, 0x3, 0x2, 0x2, 0x2, 0x18c, 0x5f, 0x3, 0x2, 
    0x2, 0x2, 0x18d, 0x18e, 0x8, 0x31, 0x1, 0x2, 0x18e, 0x18f, 0x7, 0x71, 
    0x2, 0x2, 0x18f, 0x250, 0x5, 0x60, 0x31, 0x3b, 0x190, 0x191, 0x7, 0x63, 
    0x2, 0x2, 0x191, 0x250, 0x5, 0x60, 0x31, 0x3a, 0x192, 0x193, 0x7, 0x4b, 
    0x2, 0x2, 0x193, 0x194, 0x7, 0x6b, 0x2, 0x2, 0x194, 0x195, 0x5, 0x60, 
    0x31, 0x2, 0x195, 0x196, 0x7, 0x6c, 0x2, 0x2, 0x196, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x197, 0x198, 0x7, 0x4c, 0x2, 0x2, 0x198, 0x199, 0x7, 0x6b, 
    0x2, 0x2, 0x199, 0x19a, 0x5, 0x60, 0x31, 0x2, 0x19a, 0x19b, 0x7, 0x6c, 
    0x2, 0x2, 0x19b, 0x250, 0x3, 0x2, 0x2, 0x2, 0x19c, 0x19d, 0x7, 0x4d, 
    0x2, 0x2, 0x19d, 0x19e, 0x7, 0x6b, 0x2, 0x2, 0x19e, 0x19f, 0x5, 0x60, 
    0x31, 0x2, 0x19f, 0x1a0, 0x7, 0x6c, 0x2, 0x2, 0x1a0, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1a1, 0x1a2, 0x7, 0x4e, 0x2, 0x2, 0x1a2, 0x1a3, 0x7, 0x6b, 
    0x2, 0x2, 0x1a3, 0x1a4, 0x5, 0x60, 0x31, 0x2, 0x1a4, 0x1a5, 0x7, 0x6c, 
    0x2, 0x2, 0x1a5, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1a6, 0x1a7, 0x7, 0x4f, 
    0x2, 0x2, 0x1a7, 0x1a8, 0x7, 0x6b, 0x2, 0x2, 0x1a8, 0x1a9, 0x5, 0x60, 
    0x31, 0x2, 0x1a9, 0x1aa, 0x7, 0x6c, 0x2, 0x2, 0x1aa, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1ab, 0x1ac, 0x7, 0x50, 0x2, 0x2, 0x1ac, 0x1ad, 0x7, 0x6b, 
    0x2, 0x2, 0x1ad, 0x1ae, 0x5, 0x60, 0x31, 0x2, 0x1ae, 0x1af, 0x7, 0x6c, 
    0x2, 0x2, 0x1af, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1b0, 0x1b1, 0x7, 0x51, 
    0x2, 0x2, 0x1b1, 0x1b2, 0x7, 0x6b, 0x2, 0x2, 0x1b2, 0x1b3, 0x5, 0x60, 
    0x31, 0x2, 0x1b3, 0x1b4, 0x7, 0x6c, 0x2, 0x2, 0x1b4, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1b5, 0x1b6, 0x7, 0x52, 0x2, 0x2, 0x1b6, 0x1b7, 0x7, 0x6b, 
    0x2, 0x2, 0x1b7, 0x1b8, 0x5, 0x60, 0x31, 0x2, 0x1b8, 0x1b9, 0x7, 0x6c, 
    0x2, 0x2, 0x1b9, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1ba, 0x1bb, 0x7, 0x54, 
    0x2, 0x2, 0x1bb, 0x1bc, 0x7, 0x6b, 0x2, 0x2, 0x1bc, 0x1bd, 0x5, 0x60, 
    0x31, 0x2, 0x1bd, 0x1be, 0x7, 0x6c, 0x2, 0x2, 0x1be, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1bf, 0x1c0, 0x7, 0x55, 0x2, 0x2, 0x1c0, 0x1c1, 0x7, 0x6b, 
    0x2, 0x2, 0x1c1, 0x1c2, 0x5, 0x60, 0x31, 0x2, 0x1c2, 0x1c3, 0x7, 0x6c, 
    0x2, 0x2, 0x1c3, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1c4, 0x1c5, 0x7, 0x56, 
    0x2, 0x2, 0x1c5, 0x1c6, 0x7, 0x6b, 0x2, 0x2, 0x1c6, 0x1c7, 0x5, 0x60, 
    0x31, 0x2, 0x1c7, 0x1c8, 0x7, 0x6c, 0x2, 0x2, 0x1c8, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1c9, 0x1ca, 0x7, 0x58, 0x2, 0x2, 0x1ca, 0x1cb, 0x7, 0x6b, 
    0x2, 0x2, 0x1cb, 0x1cc, 0x5, 0x60, 0x31, 0x2, 0x1cc, 0x1cd, 0x7, 0x6c, 
    0x2, 0x2, 0x1cd, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1ce, 0x1cf, 0x7, 0x59, 
    0x2, 0x2, 0x1cf, 0x1d0, 0x7, 0x6b, 0x2, 0x2, 0x1d0, 0x1d1, 0x5, 0x60, 
    0x31, 0x2, 0x1d1, 0x1d2, 0x7, 0x6c, 0x2, 0x2, 0x1d2, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1d3, 0x1d4, 0x7, 0x5a, 0x2, 0x2, 0x1d4, 0x1d5, 0x7, 0x6b, 
    0x2, 0x2, 0x1d5, 0x1d6, 0x5, 0x60, 0x31, 0x2, 0x1d6, 0x1d7, 0x7, 0x6c, 
    0x2, 0x2, 0x1d7, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1d8, 0x1d9, 0x7, 0x5c, 
    0x2, 0x2, 0x1d9, 0x1da, 0x7, 0x6b, 0x2, 0x2, 0x1da, 0x1db, 0x5, 0x60, 
    0x31, 0x2, 0x1db, 0x1dc, 0x7, 0x6c, 0x2, 0x2, 0x1dc, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1dd, 0x1de, 0x7, 0x5e, 0x2, 0x2, 0x1de, 0x1df, 0x7, 0x6b, 
    0x2, 0x2, 0x1df, 0x1e0, 0x5, 0x60, 0x31, 0x2, 0x1e0, 0x1e1, 0x7, 0x6c, 
    0x2, 0x2, 0x1e1, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1e2, 0x1e3, 0x7, 0x5d, 
    0x2, 0x2, 0x1e3, 0x1e4, 0x7, 0x6b, 0x2, 0x2, 0x1e4, 0x1e5, 0x5, 0x60, 
    0x31, 0x2, 0x1e5, 0x1e6, 0x7, 0x6c, 0x2, 0x2, 0x1e6, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1e7, 0x1e8, 0x7, 0x43, 0x2, 0x2, 0x1e8, 0x1e9, 0x7, 0x6b, 
    0x2, 0x2, 0x1e9, 0x1ea, 0x5, 0x60, 0x31, 0x2, 0x1ea, 0x1eb, 0x7, 0x6c, 
    0x2, 0x2, 0x1eb, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1ec, 0x1ed, 0x7, 0x44, 
    0x2, 0x2, 0x1ed, 0x1ee, 0x7, 0x6b, 0x2, 0x2, 0x1ee, 0x1ef, 0x5, 0x60, 
    0x31, 0x2, 0x1ef, 0x1f0, 0x7, 0x6c, 0x2, 0x2, 0x1f0, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1f1, 0x1f2, 0x7, 0x45, 0x2, 0x2, 0x1f2, 0x1f3, 0x7, 0x6b, 
    0x2, 0x2, 0x1f3, 0x1f4, 0x5, 0x60, 0x31, 0x2, 0x1f4, 0x1f5, 0x7, 0x6c, 
    0x2, 0x2, 0x1f5, 0x250, 0x3, 0x2, 0x2, 0x2, 0x1f6, 0x1f7, 0x7, 0x46, 
    0x2, 0x2, 0x1f7, 0x1f8, 0x7, 0x6b, 0x2, 0x2, 0x1f8, 0x1f9, 0x5, 0x60, 
    0x31, 0x2, 0x1f9, 0x1fa, 0x7, 0x6c, 0x2, 0x2, 0x1fa, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x1fb, 0x1fc, 0x7, 0x47, 0x2, 0x2, 0x1fc, 0x1fd, 0x7, 0x6b, 
    0x2, 0x2, 0x1fd, 0x1fe, 0x5, 0x60, 0x31, 0x2, 0x1fe, 0x1ff, 0x7, 0x6c, 
    0x2, 0x2, 0x1ff, 0x250, 0x3, 0x2, 0x2, 0x2, 0x200, 0x201, 0x7, 0x48, 
    0x2, 0x2, 0x201, 0x202, 0x7, 0x6b, 0x2, 0x2, 0x202, 0x203, 0x5, 0x60, 
    0x31, 0x2, 0x203, 0x204, 0x7, 0x6c, 0x2, 0x2, 0x204, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x205, 0x206, 0x7, 0x53, 0x2, 0x2, 0x206, 0x207, 0x7, 0x6b, 
    0x2, 0x2, 0x207, 0x208, 0x5, 0x60, 0x31, 0x2, 0x208, 0x209, 0x7, 0xd, 
    0x2, 0x2, 0x209, 0x20a, 0x5, 0x60, 0x31, 0x2, 0x20a, 0x20b, 0x7, 0x6c, 
    0x2, 0x2, 0x20b, 0x250, 0x3, 0x2, 0x2, 0x2, 0x20c, 0x20d, 0x7, 0x55, 
    0x2, 0x2, 0x20d, 0x20e, 0x7, 0x6b, 0x2, 0x2, 0x20e, 0x20f, 0x5, 0x60, 
    0x31, 0x2, 0x20f, 0x210, 0x7, 0xd, 0x2, 0x2, 0x210, 0x211, 0x5, 0x60, 
    0x31, 0x2, 0x211, 0x212, 0x7, 0x6c, 0x2, 0x2, 0x212, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x213, 0x214, 0x7, 0x57, 0x2, 0x2, 0x214, 0x215, 0x7, 0x6b, 
    0x2, 0x2, 0x215, 0x216, 0x5, 0x60, 0x31, 0x2, 0x216, 0x217, 0x7, 0xd, 
    0x2, 0x2, 0x217, 0x218, 0x5, 0x60, 0x31, 0x2, 0x218, 0x219, 0x7, 0x6c, 
    0x2, 0x2, 0x219, 0x250, 0x3, 0x2, 0x2, 0x2, 0x21a, 0x21b, 0x7, 0x5b, 
    0x2, 0x2, 0x21b, 0x21c, 0x7, 0x6b, 0x2, 0x2, 0x21c, 0x21d, 0x5, 0x60, 
    0x31, 0x2, 0x21d, 0x21e, 0x7, 0xd, 0x2, 0x2, 0x21e, 0x21f, 0x5, 0x60, 
    0x31, 0x2, 0x21f, 0x220, 0x7, 0x6c, 0x2, 0x2, 0x220, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x221, 0x222, 0x7, 0x11, 0x2, 0x2, 0x222, 0x223, 0x7, 0x6b, 
    0x2, 0x2, 0x223, 0x224, 0x5, 0x60, 0x31, 0x2, 0x224, 0x225, 0x7, 0xd, 
    0x2, 0x2, 0x225, 0x226, 0x5, 0x60, 0x31, 0x2, 0x226, 0x227, 0x7, 0x6c, 
    0x2, 0x2, 0x227, 0x250, 0x3, 0x2, 0x2, 0x2, 0x228, 0x229, 0x7, 0x5f, 
    0x2, 0x2, 0x229, 0x22a, 0x7, 0x6b, 0x2, 0x2, 0x22a, 0x22b, 0x5, 0x60, 
    0x31, 0x2, 0x22b, 0x22c, 0x7, 0xd, 0x2, 0x2, 0x22c, 0x22d, 0x5, 0x60, 
    0x31, 0x2, 0x22d, 0x22e, 0x7, 0x6c, 0x2, 0x2, 0x22e, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x22f, 0x230, 0x7, 0x60, 0x2, 0x2, 0x230, 0x231, 0x7, 0x6b, 
    0x2, 0x2, 0x231, 0x232, 0x5, 0x60, 0x31, 0x2, 0x232, 0x233, 0x7, 0xd, 
    0x2, 0x2, 0x233, 0x234, 0x5, 0x60, 0x31, 0x2, 0x234, 0x235, 0x7, 0x6c, 
    0x2, 0x2, 0x235, 0x250, 0x3, 0x2, 0x2, 0x2, 0x236, 0x237, 0x7, 0x61, 
    0x2, 0x2, 0x237, 0x238, 0x7, 0x6b, 0x2, 0x2, 0x238, 0x239, 0x5, 0x60, 
    0x31, 0x2, 0x239, 0x23a, 0x7, 0xd, 0x2, 0x2, 0x23a, 0x23b, 0x5, 0x60, 
    0x31, 0x2, 0x23b, 0x23c, 0x7, 0x6c, 0x2, 0x2, 0x23c, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x23d, 0x23e, 0x7, 0x6b, 0x2, 0x2, 0x23e, 0x23f, 0x5, 0x60, 
    0x31, 0x2, 0x23f, 0x240, 0x7, 0x6c, 0x2, 0x2, 0x240, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x241, 0x250, 0x5, 0x44, 0x23, 0x2, 0x242, 0x250, 0x5, 0x62, 
    0x32, 0x2, 0x243, 0x250, 0x7, 0x3, 0x2, 0x2, 0x244, 0x250, 0x7, 0x78, 
    0x2, 0x2, 0x245, 0x250, 0x7, 0x4a, 0x2, 0x2, 0x246, 0x250, 0x7, 0x49, 
    0x2, 0x2, 0x247, 0x250, 0x7, 0x79, 0x2, 0x2, 0x248, 0x250, 0x7, 0x7c, 
    0x2, 0x2, 0x249, 0x250, 0x7, 0x7b, 0x2, 0x2, 0x24a, 0x24b, 0x7, 0x3d, 
    0x2, 0x2, 0x24b, 0x24c, 0x7, 0x6b, 0x2, 0x2, 0x24c, 0x24d, 0x5, 0x60, 
    0x31, 0x2, 0x24d, 0x24e, 0x7, 0x6c, 0x2, 0x2, 0x24e, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x18d, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x190, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x192, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x197, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x19c, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1a1, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1a6, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1ab, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1b0, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1b5, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1ba, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1bf, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1c4, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1c9, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1ce, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1d3, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1d8, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1dd, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1e7, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1ec, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1f1, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x1f6, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x1fb, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x200, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x205, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x20c, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x213, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x21a, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x221, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x228, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x22f, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x236, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x23d, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x241, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x242, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x243, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x244, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x245, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x246, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x247, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x248, 0x3, 0x2, 
    0x2, 0x2, 0x24f, 0x249, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x24a, 0x3, 0x2, 
    0x2, 0x2, 0x250, 0x27d, 0x3, 0x2, 0x2, 0x2, 0x251, 0x252, 0xc, 0x22, 
    0x2, 0x2, 0x252, 0x253, 0x9, 0x3, 0x2, 0x2, 0x253, 0x27c, 0x5, 0x60, 
    0x31, 0x23, 0x254, 0x255, 0xc, 0x21, 0x2, 0x2, 0x255, 0x256, 0x9, 0x4, 
    0x2, 0x2, 0x256, 0x27c, 0x5, 0x60, 0x31, 0x22, 0x257, 0x258, 0xc, 0x20, 
    0x2, 0x2, 0x258, 0x259, 0x7, 0x66, 0x2, 0x2, 0x259, 0x27c, 0x5, 0x60, 
    0x31, 0x21, 0x25a, 0x25b, 0xc, 0x1b, 0x2, 0x2, 0x25b, 0x25c, 0x7, 0x67, 
    0x2, 0x2, 0x25c, 0x27c, 0x5, 0x60, 0x31, 0x1c, 0x25d, 0x25e, 0xc, 0x1a, 
    0x2, 0x2, 0x25e, 0x25f, 0x9, 0x5, 0x2, 0x2, 0x25f, 0x27c, 0x5, 0x60, 
    0x31, 0x1b, 0x260, 0x261, 0xc, 0x19, 0x2, 0x2, 0x261, 0x262, 0x9, 0x6, 
    0x2, 0x2, 0x262, 0x27c, 0x5, 0x60, 0x31, 0x1a, 0x263, 0x264, 0xc, 0x18, 
    0x2, 0x2, 0x264, 0x265, 0x9, 0x7, 0x2, 0x2, 0x265, 0x27c, 0x5, 0x60, 
    0x31, 0x19, 0x266, 0x267, 0xc, 0x17, 0x2, 0x2, 0x267, 0x268, 0x9, 0x8, 
    0x2, 0x2, 0x268, 0x27c, 0x5, 0x60, 0x31, 0x18, 0x269, 0x26a, 0xc, 0x16, 
    0x2, 0x2, 0x26a, 0x26b, 0x9, 0x9, 0x2, 0x2, 0x26b, 0x27c, 0x5, 0x60, 
    0x31, 0x17, 0x26c, 0x26d, 0xc, 0x15, 0x2, 0x2, 0x26d, 0x26e, 0x7, 0x6a, 
    0x2, 0x2, 0x26e, 0x27c, 0x5, 0x60, 0x31, 0x16, 0x26f, 0x270, 0xc, 0x10, 
    0x2, 0x2, 0x270, 0x271, 0x7, 0x34, 0x2, 0x2, 0x271, 0x272, 0x5, 0x60, 
    0x31, 0x2, 0x272, 0x273, 0x7, 0x73, 0x2, 0x2, 0x273, 0x274, 0x5, 0x60, 
    0x31, 0x11, 0x274, 0x27c, 0x3, 0x2, 0x2, 0x2, 0x275, 0x276, 0xc, 0xf, 
    0x2, 0x2, 0x276, 0x277, 0x7, 0x73, 0x2, 0x2, 0x277, 0x27c, 0x5, 0x60, 
    0x31, 0x10, 0x278, 0x279, 0xc, 0xe, 0x2, 0x2, 0x279, 0x27a, 0x7, 0x72, 
    0x2, 0x2, 0x27a, 0x27c, 0x5, 0x60, 0x31, 0xf, 0x27b, 0x251, 0x3, 0x2, 
    0x2, 0x2, 0x27b, 0x254, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x257, 0x3, 0x2, 
    0x2, 0x2, 0x27b, 0x25a, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x25d, 0x3, 0x2, 
    0x2, 0x2, 0x27b, 0x260, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x263, 0x3, 0x2, 
    0x2, 0x2, 0x27b, 0x266, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x269, 0x3, 0x2, 
    0x2, 0x2, 0x27b, 0x26c, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x26f, 0x3, 0x2, 
    0x2, 0x2, 0x27b, 0x275, 0x3, 0x2, 0x2, 0x2, 0x27b, 0x278, 0x3, 0x2, 
    0x2, 0x2, 0x27c, 0x27f, 0x3, 0x2, 0x2, 0x2, 0x27d, 0x27b, 0x3, 0x2, 
    0x2, 0x2, 0x27d, 0x27e, 0x3, 0x2, 0x2, 0x2, 0x27e, 0x61, 0x3, 0x2, 0x2, 
    0x2, 0x27f, 0x27d, 0x3, 0x2, 0x2, 0x2, 0x280, 0x287, 0x5, 0x64, 0x33, 
    0x2, 0x281, 0x287, 0x5, 0x68, 0x35, 0x2, 0x282, 0x287, 0x5, 0x66, 0x34, 
    0x2, 0x283, 0x287, 0x5, 0x6a, 0x36, 0x2, 0x284, 0x287, 0x5, 0x6c, 0x37, 
    0x2, 0x285, 0x287, 0x5, 0x6e, 0x38, 0x2, 0x286, 0x280, 0x3, 0x2, 0x2, 
    0x2, 0x286, 0x281, 0x3, 0x2, 0x2, 0x2, 0x286, 0x282, 0x3, 0x2, 0x2, 
    0x2, 0x286, 0x283, 0x3, 0x2, 0x2, 0x2, 0x286, 0x284, 0x3, 0x2, 0x2, 
    0x2, 0x286, 0x285, 0x3, 0x2, 0x2, 0x2, 0x287, 0x63, 0x3, 0x2, 0x2, 0x2, 
    0x288, 0x289, 0x7, 0x11, 0x2, 0x2, 0x289, 0x28a, 0x7, 0x6b, 0x2, 0x2, 
    0x28a, 0x28b, 0x5, 0x76, 0x3c, 0x2, 0x28b, 0x28c, 0x7, 0x6c, 0x2, 0x2, 
    0x28c, 0x65, 0x3, 0x2, 0x2, 0x2, 0x28d, 0x28e, 0x7, 0x13, 0x2, 0x2, 
    0x28e, 0x28f, 0x5, 0x74, 0x3b, 0x2, 0x28f, 0x67, 0x3, 0x2, 0x2, 0x2, 
    0x290, 0x291, 0x7, 0x15, 0x2, 0x2, 0x291, 0x292, 0x5, 0x72, 0x3a, 0x2, 
    0x292, 0x69, 0x3, 0x2, 0x2, 0x2, 0x293, 0x294, 0x7, 0x12, 0x2, 0x2, 
    0x294, 0x295, 0x7, 0x6b, 0x2, 0x2, 0x295, 0x29a, 0x5, 0x70, 0x39, 0x2, 
    0x296, 0x297, 0x7, 0xd, 0x2, 0x2, 0x297, 0x299, 0x5, 0x70, 0x39, 0x2, 
    0x298, 0x296, 0x3, 0x2, 0x2, 0x2, 0x299, 0x29c, 0x3, 0x2, 0x2, 0x2, 
    0x29a, 0x298, 0x3, 0x2, 0x2, 0x2, 0x29a, 0x29b, 0x3, 0x2, 0x2, 0x2, 
    0x29b, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x29c, 0x29a, 0x3, 0x2, 0x2, 0x2, 
    0x29d, 0x29e, 0x7, 0x6c, 0x2, 0x2, 0x29e, 0x6b, 0x3, 0x2, 0x2, 0x2, 
    0x29f, 0x2a0, 0x7, 0x14, 0x2, 0x2, 0x2a0, 0x2a1, 0x7, 0x6b, 0x2, 0x2, 
    0x2a1, 0x2a6, 0x5, 0x74, 0x3b, 0x2, 0x2a2, 0x2a3, 0x7, 0xd, 0x2, 0x2, 
    0x2a3, 0x2a5, 0x5, 0x74, 0x3b, 0x2, 0x2a4, 0x2a2, 0x3, 0x2, 0x2, 0x2, 
    0x2a5, 0x2a8, 0x3, 0x2, 0x2, 0x2, 0x2a6, 0x2a4, 0x3, 0x2, 0x2, 0x2, 
    0x2a6, 0x2a7, 0x3, 0x2, 0x2, 0x2, 0x2a7, 0x2a9, 0x3, 0x2, 0x2, 0x2, 
    0x2a8, 0x2a6, 0x3, 0x2, 0x2, 0x2, 0x2a9, 0x2aa, 0x7, 0x6c, 0x2, 0x2, 
    0x2aa, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x2ab, 0x2ac, 0x7, 0x16, 0x2, 0x2, 
    0x2ac, 0x2ad, 0x7, 0x6b, 0x2, 0x2, 0x2ad, 0x2b2, 0x5, 0x72, 0x3a, 0x2, 
    0x2ae, 0x2af, 0x7, 0xd, 0x2, 0x2, 0x2af, 0x2b1, 0x5, 0x72, 0x3a, 0x2, 
    0x2b0, 0x2ae, 0x3, 0x2, 0x2, 0x2, 0x2b1, 0x2b4, 0x3, 0x2, 0x2, 0x2, 
    0x2b2, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x2b2, 0x2b3, 0x3, 0x2, 0x2, 0x2, 
    0x2b3, 0x2b5, 0x3, 0x2, 0x2, 0x2, 0x2b4, 0x2b2, 0x3, 0x2, 0x2, 0x2, 
    0x2b5, 0x2b6, 0x7, 0x6c, 0x2, 0x2, 0x2b6, 0x6f, 0x3, 0x2, 0x2, 0x2, 
    0x2b7, 0x2bd, 0x5, 0x76, 0x3c, 0x2, 0x2b8, 0x2b9, 0x7, 0x6b, 0x2, 0x2, 
    0x2b9, 0x2ba, 0x5, 0x76, 0x3c, 0x2, 0x2ba, 0x2bb, 0x7, 0x6c, 0x2, 0x2, 
    0x2bb, 0x2bd, 0x3, 0x2, 0x2, 0x2, 0x2bc, 0x2b7, 0x3, 0x2, 0x2, 0x2, 
    0x2bc, 0x2b8, 0x3, 0x2, 0x2, 0x2, 0x2bd, 0x71, 0x3, 0x2, 0x2, 0x2, 0x2be, 
    0x2bf, 0x7, 0x6b, 0x2, 0x2, 0x2bf, 0x2c4, 0x5, 0x74, 0x3b, 0x2, 0x2c0, 
    0x2c1, 0x7, 0xd, 0x2, 0x2, 0x2c1, 0x2c3, 0x5, 0x74, 0x3b, 0x2, 0x2c2, 
    0x2c0, 0x3, 0x2, 0x2, 0x2, 0x2c3, 0x2c6, 0x3, 0x2, 0x2, 0x2, 0x2c4, 
    0x2c2, 0x3, 0x2, 0x2, 0x2, 0x2c4, 0x2c5, 0x3, 0x2, 0x2, 0x2, 0x2c5, 
    0x2c7, 0x3, 0x2, 0x2, 0x2, 0x2c6, 0x2c4, 0x3, 0x2, 0x2, 0x2, 0x2c7, 
    0x2c8, 0x7, 0x6c, 0x2, 0x2, 0x2c8, 0x73, 0x3, 0x2, 0x2, 0x2, 0x2c9, 
    0x2ca, 0x7, 0x6b, 0x2, 0x2, 0x2ca, 0x2cf, 0x5, 0x76, 0x3c, 0x2, 0x2cb, 
    0x2cc, 0x7, 0xd, 0x2, 0x2, 0x2cc, 0x2ce, 0x5, 0x76, 0x3c, 0x2, 0x2cd, 
    0x2cb, 0x3, 0x2, 0x2, 0x2, 0x2ce, 0x2d1, 0x3, 0x2, 0x2, 0x2, 0x2cf, 
    0x2cd, 0x3, 0x2, 0x2, 0x2, 0x2cf, 0x2d0, 0x3, 0x2, 0x2, 0x2, 0x2d0, 
    0x2d2, 0x3, 0x2, 0x2, 0x2, 0x2d1, 0x2cf, 0x3, 0x2, 0x2, 0x2, 0x2d2, 
    0x2d3, 0x7, 0x6c, 0x2, 0x2, 0x2d3, 0x75, 0x3, 0x2, 0x2, 0x2, 0x2d4, 
    0x2d5, 0x9, 0xa, 0x2, 0x2, 0x2d5, 0x2d6, 0x9, 0xa, 0x2, 0x2, 0x2d6, 
    0x77, 0x3, 0x2, 0x2, 0x2, 0x29, 0x7b, 0x89, 0x8e, 0x96, 0x9f, 0xa8, 
    0xac, 0xb0, 0xb4, 0xb8, 0xbc, 0xf1, 0xf6, 0xfd, 0x103, 0x11e, 0x124, 
    0x12d, 0x132, 0x139, 0x141, 0x149, 0x151, 0x15b, 0x162, 0x168, 0x172, 
    0x177, 0x18b, 0x24f, 0x27b, 0x27d, 0x286, 0x29a, 0x2a6, 0x2b2, 0x2bc, 
    0x2c4, 0x2cf, 
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

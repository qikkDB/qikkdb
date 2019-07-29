
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
    setState(133);
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
      setState(130);
      statement();
      setState(135);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(136);
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
    setState(147);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SELECT: {
        enterOuterAlt(_localctx, 1);
        setState(138);
        sqlSelect();
        break;
      }

      case GpuSqlParser::CREATEDB: {
        enterOuterAlt(_localctx, 2);
        setState(139);
        sqlCreateDb();
        break;
      }

      case GpuSqlParser::DROPDB: {
        enterOuterAlt(_localctx, 3);
        setState(140);
        sqlDropDb();
        break;
      }

      case GpuSqlParser::CREATETABLE: {
        enterOuterAlt(_localctx, 4);
        setState(141);
        sqlCreateTable();
        break;
      }

      case GpuSqlParser::DROPTABLE: {
        enterOuterAlt(_localctx, 5);
        setState(142);
        sqlDropTable();
        break;
      }

      case GpuSqlParser::ALTERTABLE: {
        enterOuterAlt(_localctx, 6);
        setState(143);
        sqlAlterTable();
        break;
      }

      case GpuSqlParser::CREATEINDEX: {
        enterOuterAlt(_localctx, 7);
        setState(144);
        sqlCreateIndex();
        break;
      }

      case GpuSqlParser::INSERTINTO: {
        enterOuterAlt(_localctx, 8);
        setState(145);
        sqlInsertInto();
        break;
      }

      case GpuSqlParser::SHOWDB:
      case GpuSqlParser::SHOWTB:
      case GpuSqlParser::SHOWCL: {
        enterOuterAlt(_localctx, 9);
        setState(146);
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
    setState(152);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::SHOWDB: {
        setState(149);
        showDatabases();
        break;
      }

      case GpuSqlParser::SHOWTB: {
        setState(150);
        showTables();
        break;
      }

      case GpuSqlParser::SHOWCL: {
        setState(151);
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
    setState(154);
    match(GpuSqlParser::SHOWDB);
    setState(155);
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
    setState(157);
    match(GpuSqlParser::SHOWTB);
    setState(160);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(158);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(159);
      database();
    }
    setState(162);
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
    setState(164);
    match(GpuSqlParser::SHOWCL);
    setState(165);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(166);
    table();
    setState(169);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(167);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(168);
      database();
    }
    setState(171);
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
    setState(173);
    match(GpuSqlParser::SELECT);
    setState(174);
    selectColumns();
    setState(175);
    match(GpuSqlParser::FROM);
    setState(176);
    fromTables();
    setState(178);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 48) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 48)) & ((1ULL << (GpuSqlParser::JOIN - 48))
      | (1ULL << (GpuSqlParser::INNER - 48))
      | (1ULL << (GpuSqlParser::FULLOUTER - 48))
      | (1ULL << (GpuSqlParser::LEFT - 48))
      | (1ULL << (GpuSqlParser::RIGHT - 48)))) != 0)) {
      setState(177);
      joinClauses();
    }
    setState(182);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::WHERE) {
      setState(180);
      match(GpuSqlParser::WHERE);
      setState(181);
      whereClause();
    }
    setState(186);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::GROUPBY) {
      setState(184);
      match(GpuSqlParser::GROUPBY);
      setState(185);
      groupByColumns();
    }
    setState(190);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::ORDERBY) {
      setState(188);
      match(GpuSqlParser::ORDERBY);
      setState(189);
      orderByColumns();
    }
    setState(194);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::LIMIT) {
      setState(192);
      match(GpuSqlParser::LIMIT);
      setState(193);
      limit();
    }
    setState(198);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::OFFSET) {
      setState(196);
      match(GpuSqlParser::OFFSET);
      setState(197);
      offset();
    }
    setState(200);
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
    setState(202);
    match(GpuSqlParser::CREATEDB);
    setState(203);
    database();
    setState(205);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::INTLIT) {
      setState(204);
      blockSize();
    }
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
    setState(209);
    match(GpuSqlParser::DROPDB);
    setState(210);
    database();
    setState(211);
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
    setState(213);
    match(GpuSqlParser::CREATETABLE);
    setState(214);
    table();
    setState(215);
    match(GpuSqlParser::LPAREN);
    setState(216);
    newTableEntries();
    setState(217);
    match(GpuSqlParser::RPAREN);
    setState(218);
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
    setState(220);
    match(GpuSqlParser::DROPTABLE);
    setState(221);
    table();
    setState(222);
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
    setState(224);
    match(GpuSqlParser::ALTERTABLE);
    setState(225);
    table();
    setState(226);
    alterTableEntries();
    setState(227);
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
    setState(229);
    match(GpuSqlParser::CREATEINDEX);
    setState(230);
    indexName();
    setState(231);
    match(GpuSqlParser::ON);
    setState(232);
    table();
    setState(233);
    match(GpuSqlParser::LPAREN);
    setState(234);
    indexColumns();
    setState(235);
    match(GpuSqlParser::RPAREN);
    setState(236);
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
    setState(238);
    match(GpuSqlParser::INSERTINTO);
    setState(239);
    table();
    setState(240);
    match(GpuSqlParser::LPAREN);
    setState(241);
    insertIntoColumns();
    setState(242);
    match(GpuSqlParser::RPAREN);
    setState(243);
    match(GpuSqlParser::VALUES);
    setState(244);
    match(GpuSqlParser::LPAREN);
    setState(245);
    insertIntoValues();
    setState(246);
    match(GpuSqlParser::RPAREN);
    setState(247);
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
    setState(249);
    newTableEntry();
    setState(254);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(250);
      match(GpuSqlParser::COMMA);
      setState(251);
      newTableEntry();
      setState(256);
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
    setState(259);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::DELIMID:
      case GpuSqlParser::ID: {
        setState(257);
        newTableColumn();
        break;
      }

      case GpuSqlParser::INDEX: {
        setState(258);
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
    setState(261);
    alterTableEntry();
    setState(266);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(262);
      match(GpuSqlParser::COMMA);
      setState(263);
      alterTableEntry();
      setState(268);
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
    setState(272);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::ADD: {
        setState(269);
        addColumn();
        break;
      }

      case GpuSqlParser::DROPCOLUMN: {
        setState(270);
        dropColumn();
        break;
      }

      case GpuSqlParser::ALTERCOLUMN: {
        setState(271);
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
    setState(274);
    match(GpuSqlParser::ADD);
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
    setState(278);
    match(GpuSqlParser::DROPCOLUMN);
    setState(279);
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
    setState(281);
    match(GpuSqlParser::ALTERCOLUMN);
    setState(282);
    column();
    setState(283);
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
    setState(285);
    column();
    setState(286);
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
    setState(288);
    match(GpuSqlParser::INDEX);
    setState(289);
    indexName();
    setState(290);
    match(GpuSqlParser::LPAREN);
    setState(291);
    indexColumns();
    setState(292);
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

std::vector<GpuSqlParser::SelectAllColumnsContext *> GpuSqlParser::SelectColumnsContext::selectAllColumns() {
  return getRuleContexts<GpuSqlParser::SelectAllColumnsContext>();
}

GpuSqlParser::SelectAllColumnsContext* GpuSqlParser::SelectColumnsContext::selectAllColumns(size_t i) {
  return getRuleContext<GpuSqlParser::SelectAllColumnsContext>(i);
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
    setState(296);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::DATETIMELIT:
      case GpuSqlParser::STRING:
      case GpuSqlParser::DELIMID:
      case GpuSqlParser::POINT:
      case GpuSqlParser::MULTIPOINT:
      case GpuSqlParser::LINESTRING:
      case GpuSqlParser::MULTILINESTRING:
      case GpuSqlParser::POLYGON:
      case GpuSqlParser::MULTIPOLYGON:
      case GpuSqlParser::AGG:
      case GpuSqlParser::YEAR:
      case GpuSqlParser::MONTH:
      case GpuSqlParser::DAY:
      case GpuSqlParser::HOUR:
      case GpuSqlParser::MINUTE:
      case GpuSqlParser::SECOND:
      case GpuSqlParser::NOW:
      case GpuSqlParser::PI:
      case GpuSqlParser::ABS:
      case GpuSqlParser::SIN:
      case GpuSqlParser::COS:
      case GpuSqlParser::TAN:
      case GpuSqlParser::COT:
      case GpuSqlParser::ASIN:
      case GpuSqlParser::ACOS:
      case GpuSqlParser::ATAN:
      case GpuSqlParser::ATAN2:
      case GpuSqlParser::LOG10:
      case GpuSqlParser::LOG:
      case GpuSqlParser::EXP:
      case GpuSqlParser::POW:
      case GpuSqlParser::SQRT:
      case GpuSqlParser::SQUARE:
      case GpuSqlParser::SIGN:
      case GpuSqlParser::ROOT:
      case GpuSqlParser::ROUND:
      case GpuSqlParser::CEIL:
      case GpuSqlParser::FLOOR:
      case GpuSqlParser::LTRIM:
      case GpuSqlParser::RTRIM:
      case GpuSqlParser::LOWER:
      case GpuSqlParser::UPPER:
      case GpuSqlParser::REVERSE:
      case GpuSqlParser::LEN:
      case GpuSqlParser::LEFT:
      case GpuSqlParser::RIGHT:
      case GpuSqlParser::CONCAT:
      case GpuSqlParser::CAST:
      case GpuSqlParser::GEO_CONTAINS:
      case GpuSqlParser::GEO_INTERSECT:
      case GpuSqlParser::GEO_UNION:
      case GpuSqlParser::MINUS:
      case GpuSqlParser::LPAREN:
      case GpuSqlParser::NOT:
      case GpuSqlParser::BOOLEANLIT:
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT:
      case GpuSqlParser::ID: {
        setState(294);
        selectColumn();
        break;
      }

      case GpuSqlParser::ASTERISK: {
        setState(295);
        selectAllColumns();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(305);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(298);
      match(GpuSqlParser::COMMA);
      setState(301);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case GpuSqlParser::DATETIMELIT:
        case GpuSqlParser::STRING:
        case GpuSqlParser::DELIMID:
        case GpuSqlParser::POINT:
        case GpuSqlParser::MULTIPOINT:
        case GpuSqlParser::LINESTRING:
        case GpuSqlParser::MULTILINESTRING:
        case GpuSqlParser::POLYGON:
        case GpuSqlParser::MULTIPOLYGON:
        case GpuSqlParser::AGG:
        case GpuSqlParser::YEAR:
        case GpuSqlParser::MONTH:
        case GpuSqlParser::DAY:
        case GpuSqlParser::HOUR:
        case GpuSqlParser::MINUTE:
        case GpuSqlParser::SECOND:
        case GpuSqlParser::NOW:
        case GpuSqlParser::PI:
        case GpuSqlParser::ABS:
        case GpuSqlParser::SIN:
        case GpuSqlParser::COS:
        case GpuSqlParser::TAN:
        case GpuSqlParser::COT:
        case GpuSqlParser::ASIN:
        case GpuSqlParser::ACOS:
        case GpuSqlParser::ATAN:
        case GpuSqlParser::ATAN2:
        case GpuSqlParser::LOG10:
        case GpuSqlParser::LOG:
        case GpuSqlParser::EXP:
        case GpuSqlParser::POW:
        case GpuSqlParser::SQRT:
        case GpuSqlParser::SQUARE:
        case GpuSqlParser::SIGN:
        case GpuSqlParser::ROOT:
        case GpuSqlParser::ROUND:
        case GpuSqlParser::CEIL:
        case GpuSqlParser::FLOOR:
        case GpuSqlParser::LTRIM:
        case GpuSqlParser::RTRIM:
        case GpuSqlParser::LOWER:
        case GpuSqlParser::UPPER:
        case GpuSqlParser::REVERSE:
        case GpuSqlParser::LEN:
        case GpuSqlParser::LEFT:
        case GpuSqlParser::RIGHT:
        case GpuSqlParser::CONCAT:
        case GpuSqlParser::CAST:
        case GpuSqlParser::GEO_CONTAINS:
        case GpuSqlParser::GEO_INTERSECT:
        case GpuSqlParser::GEO_UNION:
        case GpuSqlParser::MINUS:
        case GpuSqlParser::LPAREN:
        case GpuSqlParser::NOT:
        case GpuSqlParser::BOOLEANLIT:
        case GpuSqlParser::FLOATLIT:
        case GpuSqlParser::INTLIT:
        case GpuSqlParser::ID: {
          setState(299);
          selectColumn();
          break;
        }

        case GpuSqlParser::ASTERISK: {
          setState(300);
          selectAllColumns();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(307);
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
    setState(308);
    expression(0);
    setState(311);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(309);
      match(GpuSqlParser::AS);
      setState(310);
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

//----------------- SelectAllColumnsContext ------------------------------------------------------------------

GpuSqlParser::SelectAllColumnsContext::SelectAllColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SelectAllColumnsContext::ASTERISK() {
  return getToken(GpuSqlParser::ASTERISK, 0);
}


size_t GpuSqlParser::SelectAllColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleSelectAllColumns;
}

void GpuSqlParser::SelectAllColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSelectAllColumns(this);
}

void GpuSqlParser::SelectAllColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSelectAllColumns(this);
}

GpuSqlParser::SelectAllColumnsContext* GpuSqlParser::selectAllColumns() {
  SelectAllColumnsContext *_localctx = _tracker.createInstance<SelectAllColumnsContext>(_ctx, getState());
  enterRule(_localctx, 50, GpuSqlParser::RuleSelectAllColumns);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(313);
    match(GpuSqlParser::ASTERISK);
   
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
  enterRule(_localctx, 52, GpuSqlParser::RuleWhereClause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(315);
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
  enterRule(_localctx, 54, GpuSqlParser::RuleOrderByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(317);
    orderByColumn();
    setState(322);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(318);
      match(GpuSqlParser::COMMA);
      setState(319);
      orderByColumn();
      setState(324);
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

GpuSqlParser::ExpressionContext* GpuSqlParser::OrderByColumnContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
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
  enterRule(_localctx, 56, GpuSqlParser::RuleOrderByColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(325);
    expression(0);
    setState(327);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::DIR) {
      setState(326);
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
  enterRule(_localctx, 58, GpuSqlParser::RuleInsertIntoValues);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(329);
    columnValue();
    setState(334);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(330);
      match(GpuSqlParser::COMMA);
      setState(331);
      columnValue();
      setState(336);
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
  enterRule(_localctx, 60, GpuSqlParser::RuleInsertIntoColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(337);
    columnId();
    setState(342);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(338);
      match(GpuSqlParser::COMMA);
      setState(339);
      columnId();
      setState(344);
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
  enterRule(_localctx, 62, GpuSqlParser::RuleIndexColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(345);
    column();
    setState(350);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(346);
      match(GpuSqlParser::COMMA);
      setState(347);
      column();
      setState(352);
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
  enterRule(_localctx, 64, GpuSqlParser::RuleGroupByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(353);
    groupByColumn();
    setState(358);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(354);
      match(GpuSqlParser::COMMA);
      setState(355);
      groupByColumn();
      setState(360);
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
  enterRule(_localctx, 66, GpuSqlParser::RuleGroupByColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(361);
    expression(0);
   
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
    setState(363);
    fromTable();
    setState(368);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(364);
      match(GpuSqlParser::COMMA);
      setState(365);
      fromTable();
      setState(370);
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
    setState(372); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(371);
      joinClause();
      setState(374); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (((((_la - 48) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 48)) & ((1ULL << (GpuSqlParser::JOIN - 48))
      | (1ULL << (GpuSqlParser::INNER - 48))
      | (1ULL << (GpuSqlParser::FULLOUTER - 48))
      | (1ULL << (GpuSqlParser::LEFT - 48))
      | (1ULL << (GpuSqlParser::RIGHT - 48)))) != 0));
   
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

GpuSqlParser::JoinColumnLeftContext* GpuSqlParser::JoinClauseContext::joinColumnLeft() {
  return getRuleContext<GpuSqlParser::JoinColumnLeftContext>(0);
}

GpuSqlParser::JoinOperatorContext* GpuSqlParser::JoinClauseContext::joinOperator() {
  return getRuleContext<GpuSqlParser::JoinOperatorContext>(0);
}

GpuSqlParser::JoinColumnRightContext* GpuSqlParser::JoinClauseContext::joinColumnRight() {
  return getRuleContext<GpuSqlParser::JoinColumnRightContext>(0);
}

GpuSqlParser::JoinTypeContext* GpuSqlParser::JoinClauseContext::joinType() {
  return getRuleContext<GpuSqlParser::JoinTypeContext>(0);
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
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(377);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 59) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 59)) & ((1ULL << (GpuSqlParser::INNER - 59))
      | (1ULL << (GpuSqlParser::FULLOUTER - 59))
      | (1ULL << (GpuSqlParser::LEFT - 59))
      | (1ULL << (GpuSqlParser::RIGHT - 59)))) != 0)) {
      setState(376);
      joinType();
    }
    setState(379);
    match(GpuSqlParser::JOIN);
    setState(380);
    joinTable();
    setState(381);
    match(GpuSqlParser::ON);
    setState(382);
    joinColumnLeft();
    setState(383);
    joinOperator();
    setState(384);
    joinColumnRight();
   
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
    setState(386);
    table();
    setState(389);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(387);
      match(GpuSqlParser::AS);
      setState(388);
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

//----------------- JoinColumnLeftContext ------------------------------------------------------------------

GpuSqlParser::JoinColumnLeftContext::JoinColumnLeftContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ColumnIdContext* GpuSqlParser::JoinColumnLeftContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
}


size_t GpuSqlParser::JoinColumnLeftContext::getRuleIndex() const {
  return GpuSqlParser::RuleJoinColumnLeft;
}

void GpuSqlParser::JoinColumnLeftContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterJoinColumnLeft(this);
}

void GpuSqlParser::JoinColumnLeftContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitJoinColumnLeft(this);
}

GpuSqlParser::JoinColumnLeftContext* GpuSqlParser::joinColumnLeft() {
  JoinColumnLeftContext *_localctx = _tracker.createInstance<JoinColumnLeftContext>(_ctx, getState());
  enterRule(_localctx, 76, GpuSqlParser::RuleJoinColumnLeft);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(391);
    columnId();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- JoinColumnRightContext ------------------------------------------------------------------

GpuSqlParser::JoinColumnRightContext::JoinColumnRightContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ColumnIdContext* GpuSqlParser::JoinColumnRightContext::columnId() {
  return getRuleContext<GpuSqlParser::ColumnIdContext>(0);
}


size_t GpuSqlParser::JoinColumnRightContext::getRuleIndex() const {
  return GpuSqlParser::RuleJoinColumnRight;
}

void GpuSqlParser::JoinColumnRightContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterJoinColumnRight(this);
}

void GpuSqlParser::JoinColumnRightContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitJoinColumnRight(this);
}

GpuSqlParser::JoinColumnRightContext* GpuSqlParser::joinColumnRight() {
  JoinColumnRightContext *_localctx = _tracker.createInstance<JoinColumnRightContext>(_ctx, getState());
  enterRule(_localctx, 78, GpuSqlParser::RuleJoinColumnRight);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(393);
    columnId();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- JoinOperatorContext ------------------------------------------------------------------

GpuSqlParser::JoinOperatorContext::JoinOperatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::JoinOperatorContext::GREATER() {
  return getToken(GpuSqlParser::GREATER, 0);
}

tree::TerminalNode* GpuSqlParser::JoinOperatorContext::LESS() {
  return getToken(GpuSqlParser::LESS, 0);
}

tree::TerminalNode* GpuSqlParser::JoinOperatorContext::GREATEREQ() {
  return getToken(GpuSqlParser::GREATEREQ, 0);
}

tree::TerminalNode* GpuSqlParser::JoinOperatorContext::LESSEQ() {
  return getToken(GpuSqlParser::LESSEQ, 0);
}

tree::TerminalNode* GpuSqlParser::JoinOperatorContext::EQUALS() {
  return getToken(GpuSqlParser::EQUALS, 0);
}

tree::TerminalNode* GpuSqlParser::JoinOperatorContext::NOTEQUALS() {
  return getToken(GpuSqlParser::NOTEQUALS, 0);
}

tree::TerminalNode* GpuSqlParser::JoinOperatorContext::NOTEQUALS_GT_LT() {
  return getToken(GpuSqlParser::NOTEQUALS_GT_LT, 0);
}


size_t GpuSqlParser::JoinOperatorContext::getRuleIndex() const {
  return GpuSqlParser::RuleJoinOperator;
}

void GpuSqlParser::JoinOperatorContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterJoinOperator(this);
}

void GpuSqlParser::JoinOperatorContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitJoinOperator(this);
}

GpuSqlParser::JoinOperatorContext* GpuSqlParser::joinOperator() {
  JoinOperatorContext *_localctx = _tracker.createInstance<JoinOperatorContext>(_ctx, getState());
  enterRule(_localctx, 80, GpuSqlParser::RuleJoinOperator);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(395);
    _la = _input->LA(1);
    if (!(((((_la - 117) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 117)) & ((1ULL << (GpuSqlParser::EQUALS - 117))
      | (1ULL << (GpuSqlParser::NOTEQUALS - 117))
      | (1ULL << (GpuSqlParser::NOTEQUALS_GT_LT - 117))
      | (1ULL << (GpuSqlParser::GREATER - 117))
      | (1ULL << (GpuSqlParser::LESS - 117))
      | (1ULL << (GpuSqlParser::GREATEREQ - 117))
      | (1ULL << (GpuSqlParser::LESSEQ - 117)))) != 0))) {
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

//----------------- JoinTypeContext ------------------------------------------------------------------

GpuSqlParser::JoinTypeContext::JoinTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::JoinTypeContext::INNER() {
  return getToken(GpuSqlParser::INNER, 0);
}

tree::TerminalNode* GpuSqlParser::JoinTypeContext::LEFT() {
  return getToken(GpuSqlParser::LEFT, 0);
}

tree::TerminalNode* GpuSqlParser::JoinTypeContext::RIGHT() {
  return getToken(GpuSqlParser::RIGHT, 0);
}

tree::TerminalNode* GpuSqlParser::JoinTypeContext::FULLOUTER() {
  return getToken(GpuSqlParser::FULLOUTER, 0);
}


size_t GpuSqlParser::JoinTypeContext::getRuleIndex() const {
  return GpuSqlParser::RuleJoinType;
}

void GpuSqlParser::JoinTypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterJoinType(this);
}

void GpuSqlParser::JoinTypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitJoinType(this);
}

GpuSqlParser::JoinTypeContext* GpuSqlParser::joinType() {
  JoinTypeContext *_localctx = _tracker.createInstance<JoinTypeContext>(_ctx, getState());
  enterRule(_localctx, 82, GpuSqlParser::RuleJoinType);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(397);
    _la = _input->LA(1);
    if (!(((((_la - 59) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 59)) & ((1ULL << (GpuSqlParser::INNER - 59))
      | (1ULL << (GpuSqlParser::FULLOUTER - 59))
      | (1ULL << (GpuSqlParser::LEFT - 59))
      | (1ULL << (GpuSqlParser::RIGHT - 59)))) != 0))) {
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
  enterRule(_localctx, 84, GpuSqlParser::RuleFromTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(399);
    table();
    setState(402);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(400);
      match(GpuSqlParser::AS);
      setState(401);
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
  enterRule(_localctx, 86, GpuSqlParser::RuleColumnId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(409);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(404);
      column();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(405);
      table();
      setState(406);
      match(GpuSqlParser::DOT);
      setState(407);
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
  enterRule(_localctx, 88, GpuSqlParser::RuleTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(411);
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
  enterRule(_localctx, 90, GpuSqlParser::RuleColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(413);
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
  enterRule(_localctx, 92, GpuSqlParser::RuleDatabase);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(415);
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
  enterRule(_localctx, 94, GpuSqlParser::RuleAlias);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(417);
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
  enterRule(_localctx, 96, GpuSqlParser::RuleIndexName);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(419);
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
  enterRule(_localctx, 98, GpuSqlParser::RuleLimit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(421);
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
  enterRule(_localctx, 100, GpuSqlParser::RuleOffset);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(423);
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
  enterRule(_localctx, 102, GpuSqlParser::RuleBlockSize);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(425);
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
  enterRule(_localctx, 104, GpuSqlParser::RuleColumnValue);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(431);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::INTLIT: {
        setState(427);
        match(GpuSqlParser::INTLIT);
        break;
      }

      case GpuSqlParser::FLOATLIT: {
        setState(428);
        match(GpuSqlParser::FLOATLIT);
        break;
      }

      case GpuSqlParser::POINT:
      case GpuSqlParser::MULTIPOINT:
      case GpuSqlParser::LINESTRING:
      case GpuSqlParser::MULTILINESTRING:
      case GpuSqlParser::POLYGON:
      case GpuSqlParser::MULTIPOLYGON: {
        setState(429);
        geometry();
        break;
      }

      case GpuSqlParser::STRING: {
        setState(430);
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
//----------------- CastOperationContext ------------------------------------------------------------------

tree::TerminalNode* GpuSqlParser::CastOperationContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::CastOperationContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}

tree::TerminalNode* GpuSqlParser::CastOperationContext::AS() {
  return getToken(GpuSqlParser::AS, 0);
}

tree::TerminalNode* GpuSqlParser::CastOperationContext::DATATYPE() {
  return getToken(GpuSqlParser::DATATYPE, 0);
}

tree::TerminalNode* GpuSqlParser::CastOperationContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::CastOperationContext::CAST() {
  return getToken(GpuSqlParser::CAST, 0);
}

GpuSqlParser::CastOperationContext::CastOperationContext(ExpressionContext *ctx) { copyFrom(ctx); }

void GpuSqlParser::CastOperationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCastOperation(this);
}
void GpuSqlParser::CastOperationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCastOperation(this);
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
  size_t startState = 106;
  enterRecursionRule(_localctx, 106, GpuSqlParser::RuleExpression, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(685);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(434);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOT);
      setState(435);
      expression(67);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(436);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUS);
      setState(437);
      expression(66);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(438);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ABS);
      setState(439);
      match(GpuSqlParser::LPAREN);
      setState(440);
      expression(0);
      setState(441);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(443);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIN);
      setState(444);
      match(GpuSqlParser::LPAREN);
      setState(445);
      expression(0);
      setState(446);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(448);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COS);
      setState(449);
      match(GpuSqlParser::LPAREN);
      setState(450);
      expression(0);
      setState(451);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(453);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::TAN);
      setState(454);
      match(GpuSqlParser::LPAREN);
      setState(455);
      expression(0);
      setState(456);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(458);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COT);
      setState(459);
      match(GpuSqlParser::LPAREN);
      setState(460);
      expression(0);
      setState(461);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(463);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ASIN);
      setState(464);
      match(GpuSqlParser::LPAREN);
      setState(465);
      expression(0);
      setState(466);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(468);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ACOS);
      setState(469);
      match(GpuSqlParser::LPAREN);
      setState(470);
      expression(0);
      setState(471);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(473);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN);
      setState(474);
      match(GpuSqlParser::LPAREN);
      setState(475);
      expression(0);
      setState(476);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 11: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(478);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG10);
      setState(479);
      match(GpuSqlParser::LPAREN);
      setState(480);
      expression(0);
      setState(481);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 12: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(483);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(484);
      match(GpuSqlParser::LPAREN);
      setState(485);
      expression(0);
      setState(486);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 13: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(488);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::EXP);
      setState(489);
      match(GpuSqlParser::LPAREN);
      setState(490);
      expression(0);
      setState(491);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 14: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(493);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQRT);
      setState(494);
      match(GpuSqlParser::LPAREN);
      setState(495);
      expression(0);
      setState(496);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 15: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(498);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQUARE);
      setState(499);
      match(GpuSqlParser::LPAREN);
      setState(500);
      expression(0);
      setState(501);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 16: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(503);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIGN);
      setState(504);
      match(GpuSqlParser::LPAREN);
      setState(505);
      expression(0);
      setState(506);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 17: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(508);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROUND);
      setState(509);
      match(GpuSqlParser::LPAREN);
      setState(510);
      expression(0);
      setState(511);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 18: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(513);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::FLOOR);
      setState(514);
      match(GpuSqlParser::LPAREN);
      setState(515);
      expression(0);
      setState(516);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 19: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(518);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::CEIL);
      setState(519);
      match(GpuSqlParser::LPAREN);
      setState(520);
      expression(0);
      setState(521);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 20: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(523);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::YEAR);
      setState(524);
      match(GpuSqlParser::LPAREN);
      setState(525);
      expression(0);
      setState(526);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 21: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(528);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MONTH);
      setState(529);
      match(GpuSqlParser::LPAREN);
      setState(530);
      expression(0);
      setState(531);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 22: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(533);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DAY);
      setState(534);
      match(GpuSqlParser::LPAREN);
      setState(535);
      expression(0);
      setState(536);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 23: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(538);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::HOUR);
      setState(539);
      match(GpuSqlParser::LPAREN);
      setState(540);
      expression(0);
      setState(541);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 24: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(543);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUTE);
      setState(544);
      match(GpuSqlParser::LPAREN);
      setState(545);
      expression(0);
      setState(546);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 25: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(548);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SECOND);
      setState(549);
      match(GpuSqlParser::LPAREN);
      setState(550);
      expression(0);
      setState(551);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 26: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(553);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LTRIM);
      setState(554);
      match(GpuSqlParser::LPAREN);
      setState(555);
      expression(0);
      setState(556);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 27: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(558);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::RTRIM);
      setState(559);
      match(GpuSqlParser::LPAREN);
      setState(560);
      expression(0);
      setState(561);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 28: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(563);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOWER);
      setState(564);
      match(GpuSqlParser::LPAREN);
      setState(565);
      expression(0);
      setState(566);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 29: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(568);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::UPPER);
      setState(569);
      match(GpuSqlParser::LPAREN);
      setState(570);
      expression(0);
      setState(571);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 30: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(573);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::REVERSE);
      setState(574);
      match(GpuSqlParser::LPAREN);
      setState(575);
      expression(0);
      setState(576);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 31: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(578);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LEN);
      setState(579);
      match(GpuSqlParser::LPAREN);
      setState(580);
      expression(0);
      setState(581);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 32: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(583);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN2);
      setState(584);
      match(GpuSqlParser::LPAREN);
      setState(585);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(586);
      match(GpuSqlParser::COMMA);
      setState(587);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(588);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 33: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(590);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(591);
      match(GpuSqlParser::LPAREN);
      setState(592);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(593);
      match(GpuSqlParser::COMMA);
      setState(594);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(595);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 34: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(597);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POW);
      setState(598);
      match(GpuSqlParser::LPAREN);
      setState(599);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(600);
      match(GpuSqlParser::COMMA);
      setState(601);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(602);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 35: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(604);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROOT);
      setState(605);
      match(GpuSqlParser::LPAREN);
      setState(606);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(607);
      match(GpuSqlParser::COMMA);
      setState(608);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(609);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 36: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(611);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POINT);
      setState(612);
      match(GpuSqlParser::LPAREN);
      setState(613);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(614);
      match(GpuSqlParser::COMMA);
      setState(615);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(616);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 37: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(618);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_CONTAINS);
      setState(619);
      match(GpuSqlParser::LPAREN);
      setState(620);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(621);
      match(GpuSqlParser::COMMA);
      setState(622);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(623);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 38: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(625);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_INTERSECT);
      setState(626);
      match(GpuSqlParser::LPAREN);
      setState(627);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(628);
      match(GpuSqlParser::COMMA);
      setState(629);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(630);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 39: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(632);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_UNION);
      setState(633);
      match(GpuSqlParser::LPAREN);
      setState(634);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(635);
      match(GpuSqlParser::COMMA);
      setState(636);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(637);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 40: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(639);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::CONCAT);
      setState(640);
      match(GpuSqlParser::LPAREN);
      setState(641);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(642);
      match(GpuSqlParser::COMMA);
      setState(643);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(644);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 41: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(646);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LEFT);
      setState(647);
      match(GpuSqlParser::LPAREN);
      setState(648);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(649);
      match(GpuSqlParser::COMMA);
      setState(650);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(651);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 42: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(653);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::RIGHT);
      setState(654);
      match(GpuSqlParser::LPAREN);
      setState(655);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(656);
      match(GpuSqlParser::COMMA);
      setState(657);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(658);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 43: {
      _localctx = _tracker.createInstance<CastOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(660);
      dynamic_cast<CastOperationContext *>(_localctx)->op = match(GpuSqlParser::CAST);
      setState(661);
      match(GpuSqlParser::LPAREN);
      setState(662);
      expression(0);
      setState(663);
      match(GpuSqlParser::AS);
      setState(664);
      match(GpuSqlParser::DATATYPE);
      setState(665);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 44: {
      _localctx = _tracker.createInstance<ParenExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(667);
      match(GpuSqlParser::LPAREN);
      setState(668);
      expression(0);
      setState(669);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 45: {
      _localctx = _tracker.createInstance<VarReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(671);
      columnId();
      break;
    }

    case 46: {
      _localctx = _tracker.createInstance<GeoReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(672);
      geometry();
      break;
    }

    case 47: {
      _localctx = _tracker.createInstance<DateTimeLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(673);
      match(GpuSqlParser::DATETIMELIT);
      break;
    }

    case 48: {
      _localctx = _tracker.createInstance<DecimalLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(674);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 49: {
      _localctx = _tracker.createInstance<PiLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(675);
      match(GpuSqlParser::PI);
      break;
    }

    case 50: {
      _localctx = _tracker.createInstance<NowLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(676);
      match(GpuSqlParser::NOW);
      break;
    }

    case 51: {
      _localctx = _tracker.createInstance<IntLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(677);
      match(GpuSqlParser::INTLIT);
      break;
    }

    case 52: {
      _localctx = _tracker.createInstance<StringLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(678);
      match(GpuSqlParser::STRING);
      break;
    }

    case 53: {
      _localctx = _tracker.createInstance<BooleanLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(679);
      match(GpuSqlParser::BOOLEANLIT);
      break;
    }

    case 54: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(680);
      match(GpuSqlParser::AGG);
      setState(681);
      match(GpuSqlParser::LPAREN);
      setState(682);
      expression(0);
      setState(683);
      match(GpuSqlParser::RPAREN);
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(731);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 35, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(729);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(687);

          if (!(precpred(_ctx, 36))) throw FailedPredicateException(this, "precpred(_ctx, 36)");
          setState(688);
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
          setState(689);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(37);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(690);

          if (!(precpred(_ctx, 35))) throw FailedPredicateException(this, "precpred(_ctx, 35)");
          setState(691);
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
          setState(692);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(36);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(693);

          if (!(precpred(_ctx, 34))) throw FailedPredicateException(this, "precpred(_ctx, 34)");
          setState(694);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MODULO);
          setState(695);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(35);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(696);

          if (!(precpred(_ctx, 29))) throw FailedPredicateException(this, "precpred(_ctx, 29)");
          setState(697);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::XOR);
          setState(698);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(30);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(699);

          if (!(precpred(_ctx, 28))) throw FailedPredicateException(this, "precpred(_ctx, 28)");
          setState(700);
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
          setState(701);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(29);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(702);

          if (!(precpred(_ctx, 27))) throw FailedPredicateException(this, "precpred(_ctx, 27)");
          setState(703);
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
          setState(704);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(28);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(705);

          if (!(precpred(_ctx, 26))) throw FailedPredicateException(this, "precpred(_ctx, 26)");
          setState(706);
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
          setState(707);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(27);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(708);

          if (!(precpred(_ctx, 25))) throw FailedPredicateException(this, "precpred(_ctx, 25)");
          setState(709);
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
          setState(710);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(26);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(711);

          if (!(precpred(_ctx, 24))) throw FailedPredicateException(this, "precpred(_ctx, 24)");
          setState(712);
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
          setState(713);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(25);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(714);

          if (!(precpred(_ctx, 23))) throw FailedPredicateException(this, "precpred(_ctx, 23)");
          setState(715);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOTEQUALS_GT_LT);
          setState(716);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(24);
          break;
        }

        case 11: {
          auto newContext = _tracker.createInstance<TernaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(717);

          if (!(precpred(_ctx, 15))) throw FailedPredicateException(this, "precpred(_ctx, 15)");
          setState(718);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op = match(GpuSqlParser::BETWEEN);
          setState(719);
          expression(0);
          setState(720);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op2 = match(GpuSqlParser::AND);
          setState(721);
          expression(16);
          break;
        }

        case 12: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(723);

          if (!(precpred(_ctx, 14))) throw FailedPredicateException(this, "precpred(_ctx, 14)");
          setState(724);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::AND);
          setState(725);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(15);
          break;
        }

        case 13: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(726);

          if (!(precpred(_ctx, 13))) throw FailedPredicateException(this, "precpred(_ctx, 13)");
          setState(727);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::OR);
          setState(728);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(14);
          break;
        }

        } 
      }
      setState(733);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 35, _ctx);
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
  enterRule(_localctx, 108, GpuSqlParser::RuleGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(740);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::POINT: {
        setState(734);
        pointGeometry();
        break;
      }

      case GpuSqlParser::POLYGON: {
        setState(735);
        polygonGeometry();
        break;
      }

      case GpuSqlParser::LINESTRING: {
        setState(736);
        lineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOINT: {
        setState(737);
        multiPointGeometry();
        break;
      }

      case GpuSqlParser::MULTILINESTRING: {
        setState(738);
        multiLineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOLYGON: {
        setState(739);
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
  enterRule(_localctx, 110, GpuSqlParser::RulePointGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(742);
    match(GpuSqlParser::POINT);
    setState(743);
    match(GpuSqlParser::LPAREN);
    setState(744);
    point();
    setState(745);
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
  enterRule(_localctx, 112, GpuSqlParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(747);
    match(GpuSqlParser::LINESTRING);
    setState(748);
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
  enterRule(_localctx, 114, GpuSqlParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(750);
    match(GpuSqlParser::POLYGON);
    setState(751);
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
  enterRule(_localctx, 116, GpuSqlParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(753);
    match(GpuSqlParser::MULTIPOINT);
    setState(754);
    match(GpuSqlParser::LPAREN);
    setState(755);
    pointOrClosedPoint();
    setState(760);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(756);
      match(GpuSqlParser::COMMA);
      setState(757);
      pointOrClosedPoint();
      setState(762);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(763);
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
  enterRule(_localctx, 118, GpuSqlParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(765);
    match(GpuSqlParser::MULTILINESTRING);
    setState(766);
    match(GpuSqlParser::LPAREN);
    setState(767);
    lineString();
    setState(772);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(768);
      match(GpuSqlParser::COMMA);
      setState(769);
      lineString();
      setState(774);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(775);
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
  enterRule(_localctx, 120, GpuSqlParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(777);
    match(GpuSqlParser::MULTIPOLYGON);
    setState(778);
    match(GpuSqlParser::LPAREN);
    setState(779);
    polygon();
    setState(784);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(780);
      match(GpuSqlParser::COMMA);
      setState(781);
      polygon();
      setState(786);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(787);
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
  enterRule(_localctx, 122, GpuSqlParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(794);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT: {
        enterOuterAlt(_localctx, 1);
        setState(789);
        point();
        break;
      }

      case GpuSqlParser::LPAREN: {
        enterOuterAlt(_localctx, 2);
        setState(790);
        match(GpuSqlParser::LPAREN);
        setState(791);
        point();
        setState(792);
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
  enterRule(_localctx, 124, GpuSqlParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(796);
    match(GpuSqlParser::LPAREN);
    setState(797);
    lineString();
    setState(802);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(798);
      match(GpuSqlParser::COMMA);
      setState(799);
      lineString();
      setState(804);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(805);
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
  enterRule(_localctx, 126, GpuSqlParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(807);
    match(GpuSqlParser::LPAREN);
    setState(808);
    point();
    setState(813);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(809);
      match(GpuSqlParser::COMMA);
      setState(810);
      point();
      setState(815);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(816);
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
  enterRule(_localctx, 128, GpuSqlParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(818);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FLOATLIT

    || _la == GpuSqlParser::INTLIT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(819);
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
    case 53: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool GpuSqlParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 36);
    case 1: return precpred(_ctx, 35);
    case 2: return precpred(_ctx, 34);
    case 3: return precpred(_ctx, 29);
    case 4: return precpred(_ctx, 28);
    case 5: return precpred(_ctx, 27);
    case 6: return precpred(_ctx, 26);
    case 7: return precpred(_ctx, 25);
    case 8: return precpred(_ctx, 24);
    case 9: return precpred(_ctx, 23);
    case 10: return precpred(_ctx, 15);
    case 11: return precpred(_ctx, 14);
    case 12: return precpred(_ctx, 13);

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
  "selectColumn", "selectAllColumns", "whereClause", "orderByColumns", "orderByColumn", 
  "insertIntoValues", "insertIntoColumns", "indexColumns", "groupByColumns", 
  "groupByColumn", "fromTables", "joinClauses", "joinClause", "joinTable", 
  "joinColumnLeft", "joinColumnRight", "joinOperator", "joinType", "fromTable", 
  "columnId", "table", "column", "database", "alias", "indexName", "limit", 
  "offset", "blockSize", "columnValue", "expression", "geometry", "pointGeometry", 
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
  "", "", "", "", "", "", "", "", "", "", "", "'+'", "'-'", "'*'", "'/'", 
  "'%'", "'^'", "'='", "'!='", "'<>'", "'('", "')'", "'>'", "'<'", "'>='", 
  "'<='", "'!'", "", "", "'|'", "'&'", "'<<'", "'>>'"
};

std::vector<std::string> GpuSqlParser::_symbolicNames = {
  "", "DATETIMELIT", "LF", "CR", "CRLF", "WS", "SEMICOL", "SQOUTE", "DQOUTE", 
  "UNDERSCORE", "COLON", "COMMA", "DOT", "STRING", "DELIMID", "DATELIT", 
  "TIMELIT", "POINT", "MULTIPOINT", "LINESTRING", "MULTILINESTRING", "POLYGON", 
  "MULTIPOLYGON", "DATATYPE", "INTTYPE", "LONGTYPE", "DATETYPE", "FLOATTYPE", 
  "DOUBLETYPE", "STRINGTYPE", "BOOLEANTYPE", "POINTTYPE", "POLYTYPE", "INSERTINTO", 
  "CREATEDB", "DROPDB", "CREATETABLE", "DROPTABLE", "ALTERTABLE", "ADD", 
  "DROPCOLUMN", "ALTERCOLUMN", "CREATEINDEX", "INDEX", "PRIMARYKEY", "VALUES", 
  "SELECT", "FROM", "JOIN", "WHERE", "GROUPBY", "AS", "IN", "BETWEEN", "ON", 
  "ORDERBY", "DIR", "LIMIT", "OFFSET", "INNER", "FULLOUTER", "SHOWDB", "SHOWTB", 
  "SHOWCL", "AGG", "AVG", "SUM", "MIN", "MAX", "COUNT", "YEAR", "MONTH", 
  "DAY", "HOUR", "MINUTE", "SECOND", "NOW", "PI", "ABS", "SIN", "COS", "TAN", 
  "COT", "ASIN", "ACOS", "ATAN", "ATAN2", "LOG10", "LOG", "EXP", "POW", 
  "SQRT", "SQUARE", "SIGN", "ROOT", "ROUND", "CEIL", "FLOOR", "LTRIM", "RTRIM", 
  "LOWER", "UPPER", "REVERSE", "LEN", "LEFT", "RIGHT", "CONCAT", "CAST", 
  "GEO_CONTAINS", "GEO_INTERSECT", "GEO_UNION", "PLUS", "MINUS", "ASTERISK", 
  "DIVISION", "MODULO", "XOR", "EQUALS", "NOTEQUALS", "NOTEQUALS_GT_LT", 
  "LPAREN", "RPAREN", "GREATER", "LESS", "GREATEREQ", "LESSEQ", "NOT", "OR", 
  "AND", "BIT_OR", "BIT_AND", "L_SHIFT", "R_SHIFT", "BOOLEANLIT", "TRUE", 
  "FALSE", "FLOATLIT", "INTLIT", "ID"
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
    0x3, 0x8c, 0x338, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x9, 0x3c, 0x4, 0x3d, 0x9, 0x3d, 0x4, 0x3e, 0x9, 0x3e, 0x4, 0x3f, 0x9, 
    0x3f, 0x4, 0x40, 0x9, 0x40, 0x4, 0x41, 0x9, 0x41, 0x4, 0x42, 0x9, 0x42, 
    0x3, 0x2, 0x7, 0x2, 0x86, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x89, 0xb, 0x2, 
    0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x96, 0xa, 0x3, 0x3, 
    0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 0x9b, 0xa, 0x4, 0x3, 0x5, 0x3, 0x5, 
    0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0xa3, 0xa, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 
    0x7, 0xac, 0xa, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb5, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 
    0x8, 0xb9, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xbd, 0xa, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x5, 0x8, 0xc1, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 
    0xc5, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xc9, 0xa, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x5, 0x9, 0xd0, 0xa, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 
    0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 
    0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x7, 0x10, 0xff, 0xa, 0x10, 0xc, 0x10, 0xe, 
    0x10, 0x102, 0xb, 0x10, 0x3, 0x11, 0x3, 0x11, 0x5, 0x11, 0x106, 0xa, 
    0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x7, 0x12, 0x10b, 0xa, 0x12, 
    0xc, 0x12, 0xe, 0x12, 0x10e, 0xb, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
    0x5, 0x13, 0x113, 0xa, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 
    0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 
    0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x19, 0x3, 0x19, 0x5, 0x19, 0x12b, 
    0xa, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x5, 0x19, 0x130, 0xa, 0x19, 
    0x7, 0x19, 0x132, 0xa, 0x19, 0xc, 0x19, 0xe, 0x19, 0x135, 0xb, 0x19, 
    0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 0x13a, 0xa, 0x1a, 0x3, 0x1b, 
    0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x7, 
    0x1d, 0x143, 0xa, 0x1d, 0xc, 0x1d, 0xe, 0x1d, 0x146, 0xb, 0x1d, 0x3, 
    0x1e, 0x3, 0x1e, 0x5, 0x1e, 0x14a, 0xa, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 
    0x3, 0x1f, 0x7, 0x1f, 0x14f, 0xa, 0x1f, 0xc, 0x1f, 0xe, 0x1f, 0x152, 
    0xb, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x7, 0x20, 0x157, 0xa, 0x20, 
    0xc, 0x20, 0xe, 0x20, 0x15a, 0xb, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 
    0x7, 0x21, 0x15f, 0xa, 0x21, 0xc, 0x21, 0xe, 0x21, 0x162, 0xb, 0x21, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x7, 0x22, 0x167, 0xa, 0x22, 0xc, 0x22, 
    0xe, 0x22, 0x16a, 0xb, 0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x24, 0x3, 0x24, 
    0x3, 0x24, 0x7, 0x24, 0x171, 0xa, 0x24, 0xc, 0x24, 0xe, 0x24, 0x174, 
    0xb, 0x24, 0x3, 0x25, 0x6, 0x25, 0x177, 0xa, 0x25, 0xd, 0x25, 0xe, 0x25, 
    0x178, 0x3, 0x26, 0x5, 0x26, 0x17c, 0xa, 0x26, 0x3, 0x26, 0x3, 0x26, 
    0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x27, 0x3, 
    0x27, 0x3, 0x27, 0x5, 0x27, 0x188, 0xa, 0x27, 0x3, 0x28, 0x3, 0x28, 
    0x3, 0x29, 0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 
    0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x5, 0x2c, 0x195, 0xa, 0x2c, 0x3, 0x2d, 
    0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x5, 0x2d, 0x19c, 0xa, 0x2d, 
    0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x30, 0x3, 0x30, 0x3, 
    0x31, 0x3, 0x31, 0x3, 0x32, 0x3, 0x32, 0x3, 0x33, 0x3, 0x33, 0x3, 0x34, 
    0x3, 0x34, 0x3, 0x35, 0x3, 0x35, 0x3, 0x36, 0x3, 0x36, 0x3, 0x36, 0x3, 
    0x36, 0x5, 0x36, 0x1b2, 0xa, 0x36, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x5, 0x37, 0x2b0, 0xa, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 
    0x3, 0x37, 0x7, 0x37, 0x2dc, 0xa, 0x37, 0xc, 0x37, 0xe, 0x37, 0x2df, 
    0xb, 0x37, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 
    0x38, 0x5, 0x38, 0x2e7, 0xa, 0x38, 0x3, 0x39, 0x3, 0x39, 0x3, 0x39, 
    0x3, 0x39, 0x3, 0x39, 0x3, 0x3a, 0x3, 0x3a, 0x3, 0x3a, 0x3, 0x3b, 0x3, 
    0x3b, 0x3, 0x3b, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 
    0x7, 0x3c, 0x2f9, 0xa, 0x3c, 0xc, 0x3c, 0xe, 0x3c, 0x2fc, 0xb, 0x3c, 
    0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 
    0x3d, 0x7, 0x3d, 0x305, 0xa, 0x3d, 0xc, 0x3d, 0xe, 0x3d, 0x308, 0xb, 
    0x3d, 0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 
    0x3, 0x3e, 0x7, 0x3e, 0x311, 0xa, 0x3e, 0xc, 0x3e, 0xe, 0x3e, 0x314, 
    0xb, 0x3e, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3f, 0x3, 0x3f, 0x3, 0x3f, 0x3, 
    0x3f, 0x3, 0x3f, 0x5, 0x3f, 0x31d, 0xa, 0x3f, 0x3, 0x40, 0x3, 0x40, 
    0x3, 0x40, 0x3, 0x40, 0x7, 0x40, 0x323, 0xa, 0x40, 0xc, 0x40, 0xe, 0x40, 
    0x326, 0xb, 0x40, 0x3, 0x40, 0x3, 0x40, 0x3, 0x41, 0x3, 0x41, 0x3, 0x41, 
    0x3, 0x41, 0x7, 0x41, 0x32e, 0xa, 0x41, 0xc, 0x41, 0xe, 0x41, 0x331, 
    0xb, 0x41, 0x3, 0x41, 0x3, 0x41, 0x3, 0x42, 0x3, 0x42, 0x3, 0x42, 0x3, 
    0x42, 0x2, 0x3, 0x6c, 0x43, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 
    0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 
    0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 
    0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 
    0x5a, 0x5c, 0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 
    0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e, 0x80, 0x82, 0x2, 0xe, 0x4, 
    0x2, 0x31, 0x31, 0x36, 0x36, 0x4, 0x2, 0x77, 0x79, 0x7c, 0x7f, 0x4, 
    0x2, 0x3d, 0x3e, 0x6a, 0x6b, 0x4, 0x2, 0x10, 0x10, 0x8c, 0x8c, 0x3, 
    0x2, 0x73, 0x74, 0x3, 0x2, 0x71, 0x72, 0x3, 0x2, 0x83, 0x84, 0x3, 0x2, 
    0x85, 0x86, 0x3, 0x2, 0x7c, 0x7d, 0x3, 0x2, 0x7e, 0x7f, 0x3, 0x2, 0x77, 
    0x78, 0x3, 0x2, 0x8a, 0x8b, 0x2, 0x36f, 0x2, 0x87, 0x3, 0x2, 0x2, 0x2, 
    0x4, 0x95, 0x3, 0x2, 0x2, 0x2, 0x6, 0x9a, 0x3, 0x2, 0x2, 0x2, 0x8, 0x9c, 
    0x3, 0x2, 0x2, 0x2, 0xa, 0x9f, 0x3, 0x2, 0x2, 0x2, 0xc, 0xa6, 0x3, 0x2, 
    0x2, 0x2, 0xe, 0xaf, 0x3, 0x2, 0x2, 0x2, 0x10, 0xcc, 0x3, 0x2, 0x2, 
    0x2, 0x12, 0xd3, 0x3, 0x2, 0x2, 0x2, 0x14, 0xd7, 0x3, 0x2, 0x2, 0x2, 
    0x16, 0xde, 0x3, 0x2, 0x2, 0x2, 0x18, 0xe2, 0x3, 0x2, 0x2, 0x2, 0x1a, 
    0xe7, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xf0, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xfb, 
    0x3, 0x2, 0x2, 0x2, 0x20, 0x105, 0x3, 0x2, 0x2, 0x2, 0x22, 0x107, 0x3, 
    0x2, 0x2, 0x2, 0x24, 0x112, 0x3, 0x2, 0x2, 0x2, 0x26, 0x114, 0x3, 0x2, 
    0x2, 0x2, 0x28, 0x118, 0x3, 0x2, 0x2, 0x2, 0x2a, 0x11b, 0x3, 0x2, 0x2, 
    0x2, 0x2c, 0x11f, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x122, 0x3, 0x2, 0x2, 0x2, 
    0x30, 0x12a, 0x3, 0x2, 0x2, 0x2, 0x32, 0x136, 0x3, 0x2, 0x2, 0x2, 0x34, 
    0x13b, 0x3, 0x2, 0x2, 0x2, 0x36, 0x13d, 0x3, 0x2, 0x2, 0x2, 0x38, 0x13f, 
    0x3, 0x2, 0x2, 0x2, 0x3a, 0x147, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x14b, 0x3, 
    0x2, 0x2, 0x2, 0x3e, 0x153, 0x3, 0x2, 0x2, 0x2, 0x40, 0x15b, 0x3, 0x2, 
    0x2, 0x2, 0x42, 0x163, 0x3, 0x2, 0x2, 0x2, 0x44, 0x16b, 0x3, 0x2, 0x2, 
    0x2, 0x46, 0x16d, 0x3, 0x2, 0x2, 0x2, 0x48, 0x176, 0x3, 0x2, 0x2, 0x2, 
    0x4a, 0x17b, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x184, 0x3, 0x2, 0x2, 0x2, 0x4e, 
    0x189, 0x3, 0x2, 0x2, 0x2, 0x50, 0x18b, 0x3, 0x2, 0x2, 0x2, 0x52, 0x18d, 
    0x3, 0x2, 0x2, 0x2, 0x54, 0x18f, 0x3, 0x2, 0x2, 0x2, 0x56, 0x191, 0x3, 
    0x2, 0x2, 0x2, 0x58, 0x19b, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x19d, 0x3, 0x2, 
    0x2, 0x2, 0x5c, 0x19f, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x1a1, 0x3, 0x2, 0x2, 
    0x2, 0x60, 0x1a3, 0x3, 0x2, 0x2, 0x2, 0x62, 0x1a5, 0x3, 0x2, 0x2, 0x2, 
    0x64, 0x1a7, 0x3, 0x2, 0x2, 0x2, 0x66, 0x1a9, 0x3, 0x2, 0x2, 0x2, 0x68, 
    0x1ab, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x1b1, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x2af, 
    0x3, 0x2, 0x2, 0x2, 0x6e, 0x2e6, 0x3, 0x2, 0x2, 0x2, 0x70, 0x2e8, 0x3, 
    0x2, 0x2, 0x2, 0x72, 0x2ed, 0x3, 0x2, 0x2, 0x2, 0x74, 0x2f0, 0x3, 0x2, 
    0x2, 0x2, 0x76, 0x2f3, 0x3, 0x2, 0x2, 0x2, 0x78, 0x2ff, 0x3, 0x2, 0x2, 
    0x2, 0x7a, 0x30b, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x31c, 0x3, 0x2, 0x2, 0x2, 
    0x7e, 0x31e, 0x3, 0x2, 0x2, 0x2, 0x80, 0x329, 0x3, 0x2, 0x2, 0x2, 0x82, 
    0x334, 0x3, 0x2, 0x2, 0x2, 0x84, 0x86, 0x5, 0x4, 0x3, 0x2, 0x85, 0x84, 
    0x3, 0x2, 0x2, 0x2, 0x86, 0x89, 0x3, 0x2, 0x2, 0x2, 0x87, 0x85, 0x3, 
    0x2, 0x2, 0x2, 0x87, 0x88, 0x3, 0x2, 0x2, 0x2, 0x88, 0x8a, 0x3, 0x2, 
    0x2, 0x2, 0x89, 0x87, 0x3, 0x2, 0x2, 0x2, 0x8a, 0x8b, 0x7, 0x2, 0x2, 
    0x3, 0x8b, 0x3, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x96, 0x5, 0xe, 0x8, 0x2, 
    0x8d, 0x96, 0x5, 0x10, 0x9, 0x2, 0x8e, 0x96, 0x5, 0x12, 0xa, 0x2, 0x8f, 
    0x96, 0x5, 0x14, 0xb, 0x2, 0x90, 0x96, 0x5, 0x16, 0xc, 0x2, 0x91, 0x96, 
    0x5, 0x18, 0xd, 0x2, 0x92, 0x96, 0x5, 0x1a, 0xe, 0x2, 0x93, 0x96, 0x5, 
    0x1c, 0xf, 0x2, 0x94, 0x96, 0x5, 0x6, 0x4, 0x2, 0x95, 0x8c, 0x3, 0x2, 
    0x2, 0x2, 0x95, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x95, 0x8e, 0x3, 0x2, 0x2, 
    0x2, 0x95, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x95, 0x90, 0x3, 0x2, 0x2, 0x2, 
    0x95, 0x91, 0x3, 0x2, 0x2, 0x2, 0x95, 0x92, 0x3, 0x2, 0x2, 0x2, 0x95, 
    0x93, 0x3, 0x2, 0x2, 0x2, 0x95, 0x94, 0x3, 0x2, 0x2, 0x2, 0x96, 0x5, 
    0x3, 0x2, 0x2, 0x2, 0x97, 0x9b, 0x5, 0x8, 0x5, 0x2, 0x98, 0x9b, 0x5, 
    0xa, 0x6, 0x2, 0x99, 0x9b, 0x5, 0xc, 0x7, 0x2, 0x9a, 0x97, 0x3, 0x2, 
    0x2, 0x2, 0x9a, 0x98, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x99, 0x3, 0x2, 0x2, 
    0x2, 0x9b, 0x7, 0x3, 0x2, 0x2, 0x2, 0x9c, 0x9d, 0x7, 0x3f, 0x2, 0x2, 
    0x9d, 0x9e, 0x7, 0x8, 0x2, 0x2, 0x9e, 0x9, 0x3, 0x2, 0x2, 0x2, 0x9f, 
    0xa2, 0x7, 0x40, 0x2, 0x2, 0xa0, 0xa1, 0x9, 0x2, 0x2, 0x2, 0xa1, 0xa3, 
    0x5, 0x5e, 0x30, 0x2, 0xa2, 0xa0, 0x3, 0x2, 0x2, 0x2, 0xa2, 0xa3, 0x3, 
    0x2, 0x2, 0x2, 0xa3, 0xa4, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa5, 0x7, 0x8, 
    0x2, 0x2, 0xa5, 0xb, 0x3, 0x2, 0x2, 0x2, 0xa6, 0xa7, 0x7, 0x41, 0x2, 
    0x2, 0xa7, 0xa8, 0x9, 0x2, 0x2, 0x2, 0xa8, 0xab, 0x5, 0x5a, 0x2e, 0x2, 
    0xa9, 0xaa, 0x9, 0x2, 0x2, 0x2, 0xaa, 0xac, 0x5, 0x5e, 0x30, 0x2, 0xab, 
    0xa9, 0x3, 0x2, 0x2, 0x2, 0xab, 0xac, 0x3, 0x2, 0x2, 0x2, 0xac, 0xad, 
    0x3, 0x2, 0x2, 0x2, 0xad, 0xae, 0x7, 0x8, 0x2, 0x2, 0xae, 0xd, 0x3, 
    0x2, 0x2, 0x2, 0xaf, 0xb0, 0x7, 0x30, 0x2, 0x2, 0xb0, 0xb1, 0x5, 0x30, 
    0x19, 0x2, 0xb1, 0xb2, 0x7, 0x31, 0x2, 0x2, 0xb2, 0xb4, 0x5, 0x46, 0x24, 
    0x2, 0xb3, 0xb5, 0x5, 0x48, 0x25, 0x2, 0xb4, 0xb3, 0x3, 0x2, 0x2, 0x2, 
    0xb4, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xb6, 
    0xb7, 0x7, 0x33, 0x2, 0x2, 0xb7, 0xb9, 0x5, 0x36, 0x1c, 0x2, 0xb8, 0xb6, 
    0x3, 0x2, 0x2, 0x2, 0xb8, 0xb9, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xbc, 0x3, 
    0x2, 0x2, 0x2, 0xba, 0xbb, 0x7, 0x34, 0x2, 0x2, 0xbb, 0xbd, 0x5, 0x42, 
    0x22, 0x2, 0xbc, 0xba, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xbd, 0x3, 0x2, 0x2, 
    0x2, 0xbd, 0xc0, 0x3, 0x2, 0x2, 0x2, 0xbe, 0xbf, 0x7, 0x39, 0x2, 0x2, 
    0xbf, 0xc1, 0x5, 0x38, 0x1d, 0x2, 0xc0, 0xbe, 0x3, 0x2, 0x2, 0x2, 0xc0, 
    0xc1, 0x3, 0x2, 0x2, 0x2, 0xc1, 0xc4, 0x3, 0x2, 0x2, 0x2, 0xc2, 0xc3, 
    0x7, 0x3b, 0x2, 0x2, 0xc3, 0xc5, 0x5, 0x64, 0x33, 0x2, 0xc4, 0xc2, 0x3, 
    0x2, 0x2, 0x2, 0xc4, 0xc5, 0x3, 0x2, 0x2, 0x2, 0xc5, 0xc8, 0x3, 0x2, 
    0x2, 0x2, 0xc6, 0xc7, 0x7, 0x3c, 0x2, 0x2, 0xc7, 0xc9, 0x5, 0x66, 0x34, 
    0x2, 0xc8, 0xc6, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xc9, 0x3, 0x2, 0x2, 0x2, 
    0xc9, 0xca, 0x3, 0x2, 0x2, 0x2, 0xca, 0xcb, 0x7, 0x8, 0x2, 0x2, 0xcb, 
    0xf, 0x3, 0x2, 0x2, 0x2, 0xcc, 0xcd, 0x7, 0x24, 0x2, 0x2, 0xcd, 0xcf, 
    0x5, 0x5e, 0x30, 0x2, 0xce, 0xd0, 0x5, 0x68, 0x35, 0x2, 0xcf, 0xce, 
    0x3, 0x2, 0x2, 0x2, 0xcf, 0xd0, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xd1, 0x3, 
    0x2, 0x2, 0x2, 0xd1, 0xd2, 0x7, 0x8, 0x2, 0x2, 0xd2, 0x11, 0x3, 0x2, 
    0x2, 0x2, 0xd3, 0xd4, 0x7, 0x25, 0x2, 0x2, 0xd4, 0xd5, 0x5, 0x5e, 0x30, 
    0x2, 0xd5, 0xd6, 0x7, 0x8, 0x2, 0x2, 0xd6, 0x13, 0x3, 0x2, 0x2, 0x2, 
    0xd7, 0xd8, 0x7, 0x26, 0x2, 0x2, 0xd8, 0xd9, 0x5, 0x5a, 0x2e, 0x2, 0xd9, 
    0xda, 0x7, 0x7a, 0x2, 0x2, 0xda, 0xdb, 0x5, 0x1e, 0x10, 0x2, 0xdb, 0xdc, 
    0x7, 0x7b, 0x2, 0x2, 0xdc, 0xdd, 0x7, 0x8, 0x2, 0x2, 0xdd, 0x15, 0x3, 
    0x2, 0x2, 0x2, 0xde, 0xdf, 0x7, 0x27, 0x2, 0x2, 0xdf, 0xe0, 0x5, 0x5a, 
    0x2e, 0x2, 0xe0, 0xe1, 0x7, 0x8, 0x2, 0x2, 0xe1, 0x17, 0x3, 0x2, 0x2, 
    0x2, 0xe2, 0xe3, 0x7, 0x28, 0x2, 0x2, 0xe3, 0xe4, 0x5, 0x5a, 0x2e, 0x2, 
    0xe4, 0xe5, 0x5, 0x22, 0x12, 0x2, 0xe5, 0xe6, 0x7, 0x8, 0x2, 0x2, 0xe6, 
    0x19, 0x3, 0x2, 0x2, 0x2, 0xe7, 0xe8, 0x7, 0x2c, 0x2, 0x2, 0xe8, 0xe9, 
    0x5, 0x62, 0x32, 0x2, 0xe9, 0xea, 0x7, 0x38, 0x2, 0x2, 0xea, 0xeb, 0x5, 
    0x5a, 0x2e, 0x2, 0xeb, 0xec, 0x7, 0x7a, 0x2, 0x2, 0xec, 0xed, 0x5, 0x40, 
    0x21, 0x2, 0xed, 0xee, 0x7, 0x7b, 0x2, 0x2, 0xee, 0xef, 0x7, 0x8, 0x2, 
    0x2, 0xef, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xf1, 0x7, 0x23, 0x2, 0x2, 
    0xf1, 0xf2, 0x5, 0x5a, 0x2e, 0x2, 0xf2, 0xf3, 0x7, 0x7a, 0x2, 0x2, 0xf3, 
    0xf4, 0x5, 0x3e, 0x20, 0x2, 0xf4, 0xf5, 0x7, 0x7b, 0x2, 0x2, 0xf5, 0xf6, 
    0x7, 0x2f, 0x2, 0x2, 0xf6, 0xf7, 0x7, 0x7a, 0x2, 0x2, 0xf7, 0xf8, 0x5, 
    0x3c, 0x1f, 0x2, 0xf8, 0xf9, 0x7, 0x7b, 0x2, 0x2, 0xf9, 0xfa, 0x7, 0x8, 
    0x2, 0x2, 0xfa, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xfb, 0x100, 0x5, 0x20, 0x11, 
    0x2, 0xfc, 0xfd, 0x7, 0xd, 0x2, 0x2, 0xfd, 0xff, 0x5, 0x20, 0x11, 0x2, 
    0xfe, 0xfc, 0x3, 0x2, 0x2, 0x2, 0xff, 0x102, 0x3, 0x2, 0x2, 0x2, 0x100, 
    0xfe, 0x3, 0x2, 0x2, 0x2, 0x100, 0x101, 0x3, 0x2, 0x2, 0x2, 0x101, 0x1f, 
    0x3, 0x2, 0x2, 0x2, 0x102, 0x100, 0x3, 0x2, 0x2, 0x2, 0x103, 0x106, 
    0x5, 0x2c, 0x17, 0x2, 0x104, 0x106, 0x5, 0x2e, 0x18, 0x2, 0x105, 0x103, 
    0x3, 0x2, 0x2, 0x2, 0x105, 0x104, 0x3, 0x2, 0x2, 0x2, 0x106, 0x21, 0x3, 
    0x2, 0x2, 0x2, 0x107, 0x10c, 0x5, 0x24, 0x13, 0x2, 0x108, 0x109, 0x7, 
    0xd, 0x2, 0x2, 0x109, 0x10b, 0x5, 0x24, 0x13, 0x2, 0x10a, 0x108, 0x3, 
    0x2, 0x2, 0x2, 0x10b, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10a, 0x3, 
    0x2, 0x2, 0x2, 0x10c, 0x10d, 0x3, 0x2, 0x2, 0x2, 0x10d, 0x23, 0x3, 0x2, 
    0x2, 0x2, 0x10e, 0x10c, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x113, 0x5, 0x26, 
    0x14, 0x2, 0x110, 0x113, 0x5, 0x28, 0x15, 0x2, 0x111, 0x113, 0x5, 0x2a, 
    0x16, 0x2, 0x112, 0x10f, 0x3, 0x2, 0x2, 0x2, 0x112, 0x110, 0x3, 0x2, 
    0x2, 0x2, 0x112, 0x111, 0x3, 0x2, 0x2, 0x2, 0x113, 0x25, 0x3, 0x2, 0x2, 
    0x2, 0x114, 0x115, 0x7, 0x29, 0x2, 0x2, 0x115, 0x116, 0x5, 0x5c, 0x2f, 
    0x2, 0x116, 0x117, 0x7, 0x19, 0x2, 0x2, 0x117, 0x27, 0x3, 0x2, 0x2, 
    0x2, 0x118, 0x119, 0x7, 0x2a, 0x2, 0x2, 0x119, 0x11a, 0x5, 0x5c, 0x2f, 
    0x2, 0x11a, 0x29, 0x3, 0x2, 0x2, 0x2, 0x11b, 0x11c, 0x7, 0x2b, 0x2, 
    0x2, 0x11c, 0x11d, 0x5, 0x5c, 0x2f, 0x2, 0x11d, 0x11e, 0x7, 0x19, 0x2, 
    0x2, 0x11e, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x11f, 0x120, 0x5, 0x5c, 0x2f, 
    0x2, 0x120, 0x121, 0x7, 0x19, 0x2, 0x2, 0x121, 0x2d, 0x3, 0x2, 0x2, 
    0x2, 0x122, 0x123, 0x7, 0x2d, 0x2, 0x2, 0x123, 0x124, 0x5, 0x62, 0x32, 
    0x2, 0x124, 0x125, 0x7, 0x7a, 0x2, 0x2, 0x125, 0x126, 0x5, 0x40, 0x21, 
    0x2, 0x126, 0x127, 0x7, 0x7b, 0x2, 0x2, 0x127, 0x2f, 0x3, 0x2, 0x2, 
    0x2, 0x128, 0x12b, 0x5, 0x32, 0x1a, 0x2, 0x129, 0x12b, 0x5, 0x34, 0x1b, 
    0x2, 0x12a, 0x128, 0x3, 0x2, 0x2, 0x2, 0x12a, 0x129, 0x3, 0x2, 0x2, 
    0x2, 0x12b, 0x133, 0x3, 0x2, 0x2, 0x2, 0x12c, 0x12f, 0x7, 0xd, 0x2, 
    0x2, 0x12d, 0x130, 0x5, 0x32, 0x1a, 0x2, 0x12e, 0x130, 0x5, 0x34, 0x1b, 
    0x2, 0x12f, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x12f, 0x12e, 0x3, 0x2, 0x2, 
    0x2, 0x130, 0x132, 0x3, 0x2, 0x2, 0x2, 0x131, 0x12c, 0x3, 0x2, 0x2, 
    0x2, 0x132, 0x135, 0x3, 0x2, 0x2, 0x2, 0x133, 0x131, 0x3, 0x2, 0x2, 
    0x2, 0x133, 0x134, 0x3, 0x2, 0x2, 0x2, 0x134, 0x31, 0x3, 0x2, 0x2, 0x2, 
    0x135, 0x133, 0x3, 0x2, 0x2, 0x2, 0x136, 0x139, 0x5, 0x6c, 0x37, 0x2, 
    0x137, 0x138, 0x7, 0x35, 0x2, 0x2, 0x138, 0x13a, 0x5, 0x60, 0x31, 0x2, 
    0x139, 0x137, 0x3, 0x2, 0x2, 0x2, 0x139, 0x13a, 0x3, 0x2, 0x2, 0x2, 
    0x13a, 0x33, 0x3, 0x2, 0x2, 0x2, 0x13b, 0x13c, 0x7, 0x73, 0x2, 0x2, 
    0x13c, 0x35, 0x3, 0x2, 0x2, 0x2, 0x13d, 0x13e, 0x5, 0x6c, 0x37, 0x2, 
    0x13e, 0x37, 0x3, 0x2, 0x2, 0x2, 0x13f, 0x144, 0x5, 0x3a, 0x1e, 0x2, 
    0x140, 0x141, 0x7, 0xd, 0x2, 0x2, 0x141, 0x143, 0x5, 0x3a, 0x1e, 0x2, 
    0x142, 0x140, 0x3, 0x2, 0x2, 0x2, 0x143, 0x146, 0x3, 0x2, 0x2, 0x2, 
    0x144, 0x142, 0x3, 0x2, 0x2, 0x2, 0x144, 0x145, 0x3, 0x2, 0x2, 0x2, 
    0x145, 0x39, 0x3, 0x2, 0x2, 0x2, 0x146, 0x144, 0x3, 0x2, 0x2, 0x2, 0x147, 
    0x149, 0x5, 0x6c, 0x37, 0x2, 0x148, 0x14a, 0x7, 0x3a, 0x2, 0x2, 0x149, 
    0x148, 0x3, 0x2, 0x2, 0x2, 0x149, 0x14a, 0x3, 0x2, 0x2, 0x2, 0x14a, 
    0x3b, 0x3, 0x2, 0x2, 0x2, 0x14b, 0x150, 0x5, 0x6a, 0x36, 0x2, 0x14c, 
    0x14d, 0x7, 0xd, 0x2, 0x2, 0x14d, 0x14f, 0x5, 0x6a, 0x36, 0x2, 0x14e, 
    0x14c, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x152, 0x3, 0x2, 0x2, 0x2, 0x150, 
    0x14e, 0x3, 0x2, 0x2, 0x2, 0x150, 0x151, 0x3, 0x2, 0x2, 0x2, 0x151, 
    0x3d, 0x3, 0x2, 0x2, 0x2, 0x152, 0x150, 0x3, 0x2, 0x2, 0x2, 0x153, 0x158, 
    0x5, 0x58, 0x2d, 0x2, 0x154, 0x155, 0x7, 0xd, 0x2, 0x2, 0x155, 0x157, 
    0x5, 0x58, 0x2d, 0x2, 0x156, 0x154, 0x3, 0x2, 0x2, 0x2, 0x157, 0x15a, 
    0x3, 0x2, 0x2, 0x2, 0x158, 0x156, 0x3, 0x2, 0x2, 0x2, 0x158, 0x159, 
    0x3, 0x2, 0x2, 0x2, 0x159, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x15a, 0x158, 0x3, 
    0x2, 0x2, 0x2, 0x15b, 0x160, 0x5, 0x5c, 0x2f, 0x2, 0x15c, 0x15d, 0x7, 
    0xd, 0x2, 0x2, 0x15d, 0x15f, 0x5, 0x5c, 0x2f, 0x2, 0x15e, 0x15c, 0x3, 
    0x2, 0x2, 0x2, 0x15f, 0x162, 0x3, 0x2, 0x2, 0x2, 0x160, 0x15e, 0x3, 
    0x2, 0x2, 0x2, 0x160, 0x161, 0x3, 0x2, 0x2, 0x2, 0x161, 0x41, 0x3, 0x2, 
    0x2, 0x2, 0x162, 0x160, 0x3, 0x2, 0x2, 0x2, 0x163, 0x168, 0x5, 0x44, 
    0x23, 0x2, 0x164, 0x165, 0x7, 0xd, 0x2, 0x2, 0x165, 0x167, 0x5, 0x44, 
    0x23, 0x2, 0x166, 0x164, 0x3, 0x2, 0x2, 0x2, 0x167, 0x16a, 0x3, 0x2, 
    0x2, 0x2, 0x168, 0x166, 0x3, 0x2, 0x2, 0x2, 0x168, 0x169, 0x3, 0x2, 
    0x2, 0x2, 0x169, 0x43, 0x3, 0x2, 0x2, 0x2, 0x16a, 0x168, 0x3, 0x2, 0x2, 
    0x2, 0x16b, 0x16c, 0x5, 0x6c, 0x37, 0x2, 0x16c, 0x45, 0x3, 0x2, 0x2, 
    0x2, 0x16d, 0x172, 0x5, 0x56, 0x2c, 0x2, 0x16e, 0x16f, 0x7, 0xd, 0x2, 
    0x2, 0x16f, 0x171, 0x5, 0x56, 0x2c, 0x2, 0x170, 0x16e, 0x3, 0x2, 0x2, 
    0x2, 0x171, 0x174, 0x3, 0x2, 0x2, 0x2, 0x172, 0x170, 0x3, 0x2, 0x2, 
    0x2, 0x172, 0x173, 0x3, 0x2, 0x2, 0x2, 0x173, 0x47, 0x3, 0x2, 0x2, 0x2, 
    0x174, 0x172, 0x3, 0x2, 0x2, 0x2, 0x175, 0x177, 0x5, 0x4a, 0x26, 0x2, 
    0x176, 0x175, 0x3, 0x2, 0x2, 0x2, 0x177, 0x178, 0x3, 0x2, 0x2, 0x2, 
    0x178, 0x176, 0x3, 0x2, 0x2, 0x2, 0x178, 0x179, 0x3, 0x2, 0x2, 0x2, 
    0x179, 0x49, 0x3, 0x2, 0x2, 0x2, 0x17a, 0x17c, 0x5, 0x54, 0x2b, 0x2, 
    0x17b, 0x17a, 0x3, 0x2, 0x2, 0x2, 0x17b, 0x17c, 0x3, 0x2, 0x2, 0x2, 
    0x17c, 0x17d, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x17e, 0x7, 0x32, 0x2, 0x2, 
    0x17e, 0x17f, 0x5, 0x4c, 0x27, 0x2, 0x17f, 0x180, 0x7, 0x38, 0x2, 0x2, 
    0x180, 0x181, 0x5, 0x4e, 0x28, 0x2, 0x181, 0x182, 0x5, 0x52, 0x2a, 0x2, 
    0x182, 0x183, 0x5, 0x50, 0x29, 0x2, 0x183, 0x4b, 0x3, 0x2, 0x2, 0x2, 
    0x184, 0x187, 0x5, 0x5a, 0x2e, 0x2, 0x185, 0x186, 0x7, 0x35, 0x2, 0x2, 
    0x186, 0x188, 0x5, 0x60, 0x31, 0x2, 0x187, 0x185, 0x3, 0x2, 0x2, 0x2, 
    0x187, 0x188, 0x3, 0x2, 0x2, 0x2, 0x188, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x189, 
    0x18a, 0x5, 0x58, 0x2d, 0x2, 0x18a, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x18b, 
    0x18c, 0x5, 0x58, 0x2d, 0x2, 0x18c, 0x51, 0x3, 0x2, 0x2, 0x2, 0x18d, 
    0x18e, 0x9, 0x3, 0x2, 0x2, 0x18e, 0x53, 0x3, 0x2, 0x2, 0x2, 0x18f, 0x190, 
    0x9, 0x4, 0x2, 0x2, 0x190, 0x55, 0x3, 0x2, 0x2, 0x2, 0x191, 0x194, 0x5, 
    0x5a, 0x2e, 0x2, 0x192, 0x193, 0x7, 0x35, 0x2, 0x2, 0x193, 0x195, 0x5, 
    0x60, 0x31, 0x2, 0x194, 0x192, 0x3, 0x2, 0x2, 0x2, 0x194, 0x195, 0x3, 
    0x2, 0x2, 0x2, 0x195, 0x57, 0x3, 0x2, 0x2, 0x2, 0x196, 0x19c, 0x5, 0x5c, 
    0x2f, 0x2, 0x197, 0x198, 0x5, 0x5a, 0x2e, 0x2, 0x198, 0x199, 0x7, 0xe, 
    0x2, 0x2, 0x199, 0x19a, 0x5, 0x5c, 0x2f, 0x2, 0x19a, 0x19c, 0x3, 0x2, 
    0x2, 0x2, 0x19b, 0x196, 0x3, 0x2, 0x2, 0x2, 0x19b, 0x197, 0x3, 0x2, 
    0x2, 0x2, 0x19c, 0x59, 0x3, 0x2, 0x2, 0x2, 0x19d, 0x19e, 0x9, 0x5, 0x2, 
    0x2, 0x19e, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x19f, 0x1a0, 0x9, 0x5, 0x2, 0x2, 
    0x1a0, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x1a1, 0x1a2, 0x9, 0x5, 0x2, 0x2, 0x1a2, 
    0x5f, 0x3, 0x2, 0x2, 0x2, 0x1a3, 0x1a4, 0x9, 0x5, 0x2, 0x2, 0x1a4, 0x61, 
    0x3, 0x2, 0x2, 0x2, 0x1a5, 0x1a6, 0x9, 0x5, 0x2, 0x2, 0x1a6, 0x63, 0x3, 
    0x2, 0x2, 0x2, 0x1a7, 0x1a8, 0x7, 0x8b, 0x2, 0x2, 0x1a8, 0x65, 0x3, 
    0x2, 0x2, 0x2, 0x1a9, 0x1aa, 0x7, 0x8b, 0x2, 0x2, 0x1aa, 0x67, 0x3, 
    0x2, 0x2, 0x2, 0x1ab, 0x1ac, 0x7, 0x8b, 0x2, 0x2, 0x1ac, 0x69, 0x3, 
    0x2, 0x2, 0x2, 0x1ad, 0x1b2, 0x7, 0x8b, 0x2, 0x2, 0x1ae, 0x1b2, 0x7, 
    0x8a, 0x2, 0x2, 0x1af, 0x1b2, 0x5, 0x6e, 0x38, 0x2, 0x1b0, 0x1b2, 0x7, 
    0xf, 0x2, 0x2, 0x1b1, 0x1ad, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x1ae, 0x3, 
    0x2, 0x2, 0x2, 0x1b1, 0x1af, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x1b0, 0x3, 
    0x2, 0x2, 0x2, 0x1b2, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x1b3, 0x1b4, 0x8, 0x37, 
    0x1, 0x2, 0x1b4, 0x1b5, 0x7, 0x80, 0x2, 0x2, 0x1b5, 0x2b0, 0x5, 0x6c, 
    0x37, 0x45, 0x1b6, 0x1b7, 0x7, 0x72, 0x2, 0x2, 0x1b7, 0x2b0, 0x5, 0x6c, 
    0x37, 0x44, 0x1b8, 0x1b9, 0x7, 0x50, 0x2, 0x2, 0x1b9, 0x1ba, 0x7, 0x7a, 
    0x2, 0x2, 0x1ba, 0x1bb, 0x5, 0x6c, 0x37, 0x2, 0x1bb, 0x1bc, 0x7, 0x7b, 
    0x2, 0x2, 0x1bc, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1be, 0x7, 0x51, 
    0x2, 0x2, 0x1be, 0x1bf, 0x7, 0x7a, 0x2, 0x2, 0x1bf, 0x1c0, 0x5, 0x6c, 
    0x37, 0x2, 0x1c0, 0x1c1, 0x7, 0x7b, 0x2, 0x2, 0x1c1, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x1c2, 0x1c3, 0x7, 0x52, 0x2, 0x2, 0x1c3, 0x1c4, 0x7, 0x7a, 
    0x2, 0x2, 0x1c4, 0x1c5, 0x5, 0x6c, 0x37, 0x2, 0x1c5, 0x1c6, 0x7, 0x7b, 
    0x2, 0x2, 0x1c6, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1c8, 0x7, 0x53, 
    0x2, 0x2, 0x1c8, 0x1c9, 0x7, 0x7a, 0x2, 0x2, 0x1c9, 0x1ca, 0x5, 0x6c, 
    0x37, 0x2, 0x1ca, 0x1cb, 0x7, 0x7b, 0x2, 0x2, 0x1cb, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x1cc, 0x1cd, 0x7, 0x54, 0x2, 0x2, 0x1cd, 0x1ce, 0x7, 0x7a, 
    0x2, 0x2, 0x1ce, 0x1cf, 0x5, 0x6c, 0x37, 0x2, 0x1cf, 0x1d0, 0x7, 0x7b, 
    0x2, 0x2, 0x1d0, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x1d1, 0x1d2, 0x7, 0x55, 
    0x2, 0x2, 0x1d2, 0x1d3, 0x7, 0x7a, 0x2, 0x2, 0x1d3, 0x1d4, 0x5, 0x6c, 
    0x37, 0x2, 0x1d4, 0x1d5, 0x7, 0x7b, 0x2, 0x2, 0x1d5, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x1d6, 0x1d7, 0x7, 0x56, 0x2, 0x2, 0x1d7, 0x1d8, 0x7, 0x7a, 
    0x2, 0x2, 0x1d8, 0x1d9, 0x5, 0x6c, 0x37, 0x2, 0x1d9, 0x1da, 0x7, 0x7b, 
    0x2, 0x2, 0x1da, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x1db, 0x1dc, 0x7, 0x57, 
    0x2, 0x2, 0x1dc, 0x1dd, 0x7, 0x7a, 0x2, 0x2, 0x1dd, 0x1de, 0x5, 0x6c, 
    0x37, 0x2, 0x1de, 0x1df, 0x7, 0x7b, 0x2, 0x2, 0x1df, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x1e0, 0x1e1, 0x7, 0x59, 0x2, 0x2, 0x1e1, 0x1e2, 0x7, 0x7a, 
    0x2, 0x2, 0x1e2, 0x1e3, 0x5, 0x6c, 0x37, 0x2, 0x1e3, 0x1e4, 0x7, 0x7b, 
    0x2, 0x2, 0x1e4, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x1e5, 0x1e6, 0x7, 0x5a, 
    0x2, 0x2, 0x1e6, 0x1e7, 0x7, 0x7a, 0x2, 0x2, 0x1e7, 0x1e8, 0x5, 0x6c, 
    0x37, 0x2, 0x1e8, 0x1e9, 0x7, 0x7b, 0x2, 0x2, 0x1e9, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x1ea, 0x1eb, 0x7, 0x5b, 0x2, 0x2, 0x1eb, 0x1ec, 0x7, 0x7a, 
    0x2, 0x2, 0x1ec, 0x1ed, 0x5, 0x6c, 0x37, 0x2, 0x1ed, 0x1ee, 0x7, 0x7b, 
    0x2, 0x2, 0x1ee, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x1ef, 0x1f0, 0x7, 0x5d, 
    0x2, 0x2, 0x1f0, 0x1f1, 0x7, 0x7a, 0x2, 0x2, 0x1f1, 0x1f2, 0x5, 0x6c, 
    0x37, 0x2, 0x1f2, 0x1f3, 0x7, 0x7b, 0x2, 0x2, 0x1f3, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x1f4, 0x1f5, 0x7, 0x5e, 0x2, 0x2, 0x1f5, 0x1f6, 0x7, 0x7a, 
    0x2, 0x2, 0x1f6, 0x1f7, 0x5, 0x6c, 0x37, 0x2, 0x1f7, 0x1f8, 0x7, 0x7b, 
    0x2, 0x2, 0x1f8, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x1f9, 0x1fa, 0x7, 0x5f, 
    0x2, 0x2, 0x1fa, 0x1fb, 0x7, 0x7a, 0x2, 0x2, 0x1fb, 0x1fc, 0x5, 0x6c, 
    0x37, 0x2, 0x1fc, 0x1fd, 0x7, 0x7b, 0x2, 0x2, 0x1fd, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x1fe, 0x1ff, 0x7, 0x61, 0x2, 0x2, 0x1ff, 0x200, 0x7, 0x7a, 
    0x2, 0x2, 0x200, 0x201, 0x5, 0x6c, 0x37, 0x2, 0x201, 0x202, 0x7, 0x7b, 
    0x2, 0x2, 0x202, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x203, 0x204, 0x7, 0x63, 
    0x2, 0x2, 0x204, 0x205, 0x7, 0x7a, 0x2, 0x2, 0x205, 0x206, 0x5, 0x6c, 
    0x37, 0x2, 0x206, 0x207, 0x7, 0x7b, 0x2, 0x2, 0x207, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x208, 0x209, 0x7, 0x62, 0x2, 0x2, 0x209, 0x20a, 0x7, 0x7a, 
    0x2, 0x2, 0x20a, 0x20b, 0x5, 0x6c, 0x37, 0x2, 0x20b, 0x20c, 0x7, 0x7b, 
    0x2, 0x2, 0x20c, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x20d, 0x20e, 0x7, 0x48, 
    0x2, 0x2, 0x20e, 0x20f, 0x7, 0x7a, 0x2, 0x2, 0x20f, 0x210, 0x5, 0x6c, 
    0x37, 0x2, 0x210, 0x211, 0x7, 0x7b, 0x2, 0x2, 0x211, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x212, 0x213, 0x7, 0x49, 0x2, 0x2, 0x213, 0x214, 0x7, 0x7a, 
    0x2, 0x2, 0x214, 0x215, 0x5, 0x6c, 0x37, 0x2, 0x215, 0x216, 0x7, 0x7b, 
    0x2, 0x2, 0x216, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x217, 0x218, 0x7, 0x4a, 
    0x2, 0x2, 0x218, 0x219, 0x7, 0x7a, 0x2, 0x2, 0x219, 0x21a, 0x5, 0x6c, 
    0x37, 0x2, 0x21a, 0x21b, 0x7, 0x7b, 0x2, 0x2, 0x21b, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x21c, 0x21d, 0x7, 0x4b, 0x2, 0x2, 0x21d, 0x21e, 0x7, 0x7a, 
    0x2, 0x2, 0x21e, 0x21f, 0x5, 0x6c, 0x37, 0x2, 0x21f, 0x220, 0x7, 0x7b, 
    0x2, 0x2, 0x220, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x221, 0x222, 0x7, 0x4c, 
    0x2, 0x2, 0x222, 0x223, 0x7, 0x7a, 0x2, 0x2, 0x223, 0x224, 0x5, 0x6c, 
    0x37, 0x2, 0x224, 0x225, 0x7, 0x7b, 0x2, 0x2, 0x225, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x226, 0x227, 0x7, 0x4d, 0x2, 0x2, 0x227, 0x228, 0x7, 0x7a, 
    0x2, 0x2, 0x228, 0x229, 0x5, 0x6c, 0x37, 0x2, 0x229, 0x22a, 0x7, 0x7b, 
    0x2, 0x2, 0x22a, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x22b, 0x22c, 0x7, 0x64, 
    0x2, 0x2, 0x22c, 0x22d, 0x7, 0x7a, 0x2, 0x2, 0x22d, 0x22e, 0x5, 0x6c, 
    0x37, 0x2, 0x22e, 0x22f, 0x7, 0x7b, 0x2, 0x2, 0x22f, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x230, 0x231, 0x7, 0x65, 0x2, 0x2, 0x231, 0x232, 0x7, 0x7a, 
    0x2, 0x2, 0x232, 0x233, 0x5, 0x6c, 0x37, 0x2, 0x233, 0x234, 0x7, 0x7b, 
    0x2, 0x2, 0x234, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x235, 0x236, 0x7, 0x66, 
    0x2, 0x2, 0x236, 0x237, 0x7, 0x7a, 0x2, 0x2, 0x237, 0x238, 0x5, 0x6c, 
    0x37, 0x2, 0x238, 0x239, 0x7, 0x7b, 0x2, 0x2, 0x239, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x23a, 0x23b, 0x7, 0x67, 0x2, 0x2, 0x23b, 0x23c, 0x7, 0x7a, 
    0x2, 0x2, 0x23c, 0x23d, 0x5, 0x6c, 0x37, 0x2, 0x23d, 0x23e, 0x7, 0x7b, 
    0x2, 0x2, 0x23e, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x23f, 0x240, 0x7, 0x68, 
    0x2, 0x2, 0x240, 0x241, 0x7, 0x7a, 0x2, 0x2, 0x241, 0x242, 0x5, 0x6c, 
    0x37, 0x2, 0x242, 0x243, 0x7, 0x7b, 0x2, 0x2, 0x243, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x244, 0x245, 0x7, 0x69, 0x2, 0x2, 0x245, 0x246, 0x7, 0x7a, 
    0x2, 0x2, 0x246, 0x247, 0x5, 0x6c, 0x37, 0x2, 0x247, 0x248, 0x7, 0x7b, 
    0x2, 0x2, 0x248, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x249, 0x24a, 0x7, 0x58, 
    0x2, 0x2, 0x24a, 0x24b, 0x7, 0x7a, 0x2, 0x2, 0x24b, 0x24c, 0x5, 0x6c, 
    0x37, 0x2, 0x24c, 0x24d, 0x7, 0xd, 0x2, 0x2, 0x24d, 0x24e, 0x5, 0x6c, 
    0x37, 0x2, 0x24e, 0x24f, 0x7, 0x7b, 0x2, 0x2, 0x24f, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x250, 0x251, 0x7, 0x5a, 0x2, 0x2, 0x251, 0x252, 0x7, 0x7a, 
    0x2, 0x2, 0x252, 0x253, 0x5, 0x6c, 0x37, 0x2, 0x253, 0x254, 0x7, 0xd, 
    0x2, 0x2, 0x254, 0x255, 0x5, 0x6c, 0x37, 0x2, 0x255, 0x256, 0x7, 0x7b, 
    0x2, 0x2, 0x256, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x257, 0x258, 0x7, 0x5c, 
    0x2, 0x2, 0x258, 0x259, 0x7, 0x7a, 0x2, 0x2, 0x259, 0x25a, 0x5, 0x6c, 
    0x37, 0x2, 0x25a, 0x25b, 0x7, 0xd, 0x2, 0x2, 0x25b, 0x25c, 0x5, 0x6c, 
    0x37, 0x2, 0x25c, 0x25d, 0x7, 0x7b, 0x2, 0x2, 0x25d, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x25e, 0x25f, 0x7, 0x60, 0x2, 0x2, 0x25f, 0x260, 0x7, 0x7a, 
    0x2, 0x2, 0x260, 0x261, 0x5, 0x6c, 0x37, 0x2, 0x261, 0x262, 0x7, 0xd, 
    0x2, 0x2, 0x262, 0x263, 0x5, 0x6c, 0x37, 0x2, 0x263, 0x264, 0x7, 0x7b, 
    0x2, 0x2, 0x264, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x265, 0x266, 0x7, 0x13, 
    0x2, 0x2, 0x266, 0x267, 0x7, 0x7a, 0x2, 0x2, 0x267, 0x268, 0x5, 0x6c, 
    0x37, 0x2, 0x268, 0x269, 0x7, 0xd, 0x2, 0x2, 0x269, 0x26a, 0x5, 0x6c, 
    0x37, 0x2, 0x26a, 0x26b, 0x7, 0x7b, 0x2, 0x2, 0x26b, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x26c, 0x26d, 0x7, 0x6e, 0x2, 0x2, 0x26d, 0x26e, 0x7, 0x7a, 
    0x2, 0x2, 0x26e, 0x26f, 0x5, 0x6c, 0x37, 0x2, 0x26f, 0x270, 0x7, 0xd, 
    0x2, 0x2, 0x270, 0x271, 0x5, 0x6c, 0x37, 0x2, 0x271, 0x272, 0x7, 0x7b, 
    0x2, 0x2, 0x272, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x273, 0x274, 0x7, 0x6f, 
    0x2, 0x2, 0x274, 0x275, 0x7, 0x7a, 0x2, 0x2, 0x275, 0x276, 0x5, 0x6c, 
    0x37, 0x2, 0x276, 0x277, 0x7, 0xd, 0x2, 0x2, 0x277, 0x278, 0x5, 0x6c, 
    0x37, 0x2, 0x278, 0x279, 0x7, 0x7b, 0x2, 0x2, 0x279, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x27a, 0x27b, 0x7, 0x70, 0x2, 0x2, 0x27b, 0x27c, 0x7, 0x7a, 
    0x2, 0x2, 0x27c, 0x27d, 0x5, 0x6c, 0x37, 0x2, 0x27d, 0x27e, 0x7, 0xd, 
    0x2, 0x2, 0x27e, 0x27f, 0x5, 0x6c, 0x37, 0x2, 0x27f, 0x280, 0x7, 0x7b, 
    0x2, 0x2, 0x280, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x281, 0x282, 0x7, 0x6c, 
    0x2, 0x2, 0x282, 0x283, 0x7, 0x7a, 0x2, 0x2, 0x283, 0x284, 0x5, 0x6c, 
    0x37, 0x2, 0x284, 0x285, 0x7, 0xd, 0x2, 0x2, 0x285, 0x286, 0x5, 0x6c, 
    0x37, 0x2, 0x286, 0x287, 0x7, 0x7b, 0x2, 0x2, 0x287, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x288, 0x289, 0x7, 0x6a, 0x2, 0x2, 0x289, 0x28a, 0x7, 0x7a, 
    0x2, 0x2, 0x28a, 0x28b, 0x5, 0x6c, 0x37, 0x2, 0x28b, 0x28c, 0x7, 0xd, 
    0x2, 0x2, 0x28c, 0x28d, 0x5, 0x6c, 0x37, 0x2, 0x28d, 0x28e, 0x7, 0x7b, 
    0x2, 0x2, 0x28e, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x28f, 0x290, 0x7, 0x6b, 
    0x2, 0x2, 0x290, 0x291, 0x7, 0x7a, 0x2, 0x2, 0x291, 0x292, 0x5, 0x6c, 
    0x37, 0x2, 0x292, 0x293, 0x7, 0xd, 0x2, 0x2, 0x293, 0x294, 0x5, 0x6c, 
    0x37, 0x2, 0x294, 0x295, 0x7, 0x7b, 0x2, 0x2, 0x295, 0x2b0, 0x3, 0x2, 
    0x2, 0x2, 0x296, 0x297, 0x7, 0x6d, 0x2, 0x2, 0x297, 0x298, 0x7, 0x7a, 
    0x2, 0x2, 0x298, 0x299, 0x5, 0x6c, 0x37, 0x2, 0x299, 0x29a, 0x7, 0x35, 
    0x2, 0x2, 0x29a, 0x29b, 0x7, 0x19, 0x2, 0x2, 0x29b, 0x29c, 0x7, 0x7b, 
    0x2, 0x2, 0x29c, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x29d, 0x29e, 0x7, 0x7a, 
    0x2, 0x2, 0x29e, 0x29f, 0x5, 0x6c, 0x37, 0x2, 0x29f, 0x2a0, 0x7, 0x7b, 
    0x2, 0x2, 0x2a0, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x2a1, 0x2b0, 0x5, 0x58, 
    0x2d, 0x2, 0x2a2, 0x2b0, 0x5, 0x6e, 0x38, 0x2, 0x2a3, 0x2b0, 0x7, 0x3, 
    0x2, 0x2, 0x2a4, 0x2b0, 0x7, 0x8a, 0x2, 0x2, 0x2a5, 0x2b0, 0x7, 0x4f, 
    0x2, 0x2, 0x2a6, 0x2b0, 0x7, 0x4e, 0x2, 0x2, 0x2a7, 0x2b0, 0x7, 0x8b, 
    0x2, 0x2, 0x2a8, 0x2b0, 0x7, 0xf, 0x2, 0x2, 0x2a9, 0x2b0, 0x7, 0x87, 
    0x2, 0x2, 0x2aa, 0x2ab, 0x7, 0x42, 0x2, 0x2, 0x2ab, 0x2ac, 0x7, 0x7a, 
    0x2, 0x2, 0x2ac, 0x2ad, 0x5, 0x6c, 0x37, 0x2, 0x2ad, 0x2ae, 0x7, 0x7b, 
    0x2, 0x2, 0x2ae, 0x2b0, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1b3, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1b6, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1b8, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1c2, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1c7, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1cc, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1d1, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1d6, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1db, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1e0, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1e5, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1ea, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1ef, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1f4, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x1f9, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x1fe, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x203, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x208, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x20d, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x212, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x217, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x21c, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x221, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x226, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x22b, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x230, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x235, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x23a, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x23f, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x244, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x249, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x250, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x257, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x25e, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x265, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x26c, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x273, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x27a, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x281, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x288, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x28f, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x296, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x29d, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x2a1, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x2a2, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x2a3, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x2a4, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x2a5, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x2a6, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x2a7, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x2a8, 0x3, 0x2, 0x2, 0x2, 0x2af, 0x2a9, 0x3, 0x2, 
    0x2, 0x2, 0x2af, 0x2aa, 0x3, 0x2, 0x2, 0x2, 0x2b0, 0x2dd, 0x3, 0x2, 
    0x2, 0x2, 0x2b1, 0x2b2, 0xc, 0x26, 0x2, 0x2, 0x2b2, 0x2b3, 0x9, 0x6, 
    0x2, 0x2, 0x2b3, 0x2dc, 0x5, 0x6c, 0x37, 0x27, 0x2b4, 0x2b5, 0xc, 0x25, 
    0x2, 0x2, 0x2b5, 0x2b6, 0x9, 0x7, 0x2, 0x2, 0x2b6, 0x2dc, 0x5, 0x6c, 
    0x37, 0x26, 0x2b7, 0x2b8, 0xc, 0x24, 0x2, 0x2, 0x2b8, 0x2b9, 0x7, 0x75, 
    0x2, 0x2, 0x2b9, 0x2dc, 0x5, 0x6c, 0x37, 0x25, 0x2ba, 0x2bb, 0xc, 0x1f, 
    0x2, 0x2, 0x2bb, 0x2bc, 0x7, 0x76, 0x2, 0x2, 0x2bc, 0x2dc, 0x5, 0x6c, 
    0x37, 0x20, 0x2bd, 0x2be, 0xc, 0x1e, 0x2, 0x2, 0x2be, 0x2bf, 0x9, 0x8, 
    0x2, 0x2, 0x2bf, 0x2dc, 0x5, 0x6c, 0x37, 0x1f, 0x2c0, 0x2c1, 0xc, 0x1d, 
    0x2, 0x2, 0x2c1, 0x2c2, 0x9, 0x9, 0x2, 0x2, 0x2c2, 0x2dc, 0x5, 0x6c, 
    0x37, 0x1e, 0x2c3, 0x2c4, 0xc, 0x1c, 0x2, 0x2, 0x2c4, 0x2c5, 0x9, 0xa, 
    0x2, 0x2, 0x2c5, 0x2dc, 0x5, 0x6c, 0x37, 0x1d, 0x2c6, 0x2c7, 0xc, 0x1b, 
    0x2, 0x2, 0x2c7, 0x2c8, 0x9, 0xb, 0x2, 0x2, 0x2c8, 0x2dc, 0x5, 0x6c, 
    0x37, 0x1c, 0x2c9, 0x2ca, 0xc, 0x1a, 0x2, 0x2, 0x2ca, 0x2cb, 0x9, 0xc, 
    0x2, 0x2, 0x2cb, 0x2dc, 0x5, 0x6c, 0x37, 0x1b, 0x2cc, 0x2cd, 0xc, 0x19, 
    0x2, 0x2, 0x2cd, 0x2ce, 0x7, 0x79, 0x2, 0x2, 0x2ce, 0x2dc, 0x5, 0x6c, 
    0x37, 0x1a, 0x2cf, 0x2d0, 0xc, 0x11, 0x2, 0x2, 0x2d0, 0x2d1, 0x7, 0x37, 
    0x2, 0x2, 0x2d1, 0x2d2, 0x5, 0x6c, 0x37, 0x2, 0x2d2, 0x2d3, 0x7, 0x82, 
    0x2, 0x2, 0x2d3, 0x2d4, 0x5, 0x6c, 0x37, 0x12, 0x2d4, 0x2dc, 0x3, 0x2, 
    0x2, 0x2, 0x2d5, 0x2d6, 0xc, 0x10, 0x2, 0x2, 0x2d6, 0x2d7, 0x7, 0x82, 
    0x2, 0x2, 0x2d7, 0x2dc, 0x5, 0x6c, 0x37, 0x11, 0x2d8, 0x2d9, 0xc, 0xf, 
    0x2, 0x2, 0x2d9, 0x2da, 0x7, 0x81, 0x2, 0x2, 0x2da, 0x2dc, 0x5, 0x6c, 
    0x37, 0x10, 0x2db, 0x2b1, 0x3, 0x2, 0x2, 0x2, 0x2db, 0x2b4, 0x3, 0x2, 
    0x2, 0x2, 0x2db, 0x2b7, 0x3, 0x2, 0x2, 0x2, 0x2db, 0x2ba, 0x3, 0x2, 
    0x2, 0x2, 0x2db, 0x2bd, 0x3, 0x2, 0x2, 0x2, 0x2db, 0x2c0, 0x3, 0x2, 
    0x2, 0x2, 0x2db, 0x2c3, 0x3, 0x2, 0x2, 0x2, 0x2db, 0x2c6, 0x3, 0x2, 
    0x2, 0x2, 0x2db, 0x2c9, 0x3, 0x2, 0x2, 0x2, 0x2db, 0x2cc, 0x3, 0x2, 
    0x2, 0x2, 0x2db, 0x2cf, 0x3, 0x2, 0x2, 0x2, 0x2db, 0x2d5, 0x3, 0x2, 
    0x2, 0x2, 0x2db, 0x2d8, 0x3, 0x2, 0x2, 0x2, 0x2dc, 0x2df, 0x3, 0x2, 
    0x2, 0x2, 0x2dd, 0x2db, 0x3, 0x2, 0x2, 0x2, 0x2dd, 0x2de, 0x3, 0x2, 
    0x2, 0x2, 0x2de, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x2df, 0x2dd, 0x3, 0x2, 0x2, 
    0x2, 0x2e0, 0x2e7, 0x5, 0x70, 0x39, 0x2, 0x2e1, 0x2e7, 0x5, 0x74, 0x3b, 
    0x2, 0x2e2, 0x2e7, 0x5, 0x72, 0x3a, 0x2, 0x2e3, 0x2e7, 0x5, 0x76, 0x3c, 
    0x2, 0x2e4, 0x2e7, 0x5, 0x78, 0x3d, 0x2, 0x2e5, 0x2e7, 0x5, 0x7a, 0x3e, 
    0x2, 0x2e6, 0x2e0, 0x3, 0x2, 0x2, 0x2, 0x2e6, 0x2e1, 0x3, 0x2, 0x2, 
    0x2, 0x2e6, 0x2e2, 0x3, 0x2, 0x2, 0x2, 0x2e6, 0x2e3, 0x3, 0x2, 0x2, 
    0x2, 0x2e6, 0x2e4, 0x3, 0x2, 0x2, 0x2, 0x2e6, 0x2e5, 0x3, 0x2, 0x2, 
    0x2, 0x2e7, 0x6f, 0x3, 0x2, 0x2, 0x2, 0x2e8, 0x2e9, 0x7, 0x13, 0x2, 
    0x2, 0x2e9, 0x2ea, 0x7, 0x7a, 0x2, 0x2, 0x2ea, 0x2eb, 0x5, 0x82, 0x42, 
    0x2, 0x2eb, 0x2ec, 0x7, 0x7b, 0x2, 0x2, 0x2ec, 0x71, 0x3, 0x2, 0x2, 
    0x2, 0x2ed, 0x2ee, 0x7, 0x15, 0x2, 0x2, 0x2ee, 0x2ef, 0x5, 0x80, 0x41, 
    0x2, 0x2ef, 0x73, 0x3, 0x2, 0x2, 0x2, 0x2f0, 0x2f1, 0x7, 0x17, 0x2, 
    0x2, 0x2f1, 0x2f2, 0x5, 0x7e, 0x40, 0x2, 0x2f2, 0x75, 0x3, 0x2, 0x2, 
    0x2, 0x2f3, 0x2f4, 0x7, 0x14, 0x2, 0x2, 0x2f4, 0x2f5, 0x7, 0x7a, 0x2, 
    0x2, 0x2f5, 0x2fa, 0x5, 0x7c, 0x3f, 0x2, 0x2f6, 0x2f7, 0x7, 0xd, 0x2, 
    0x2, 0x2f7, 0x2f9, 0x5, 0x7c, 0x3f, 0x2, 0x2f8, 0x2f6, 0x3, 0x2, 0x2, 
    0x2, 0x2f9, 0x2fc, 0x3, 0x2, 0x2, 0x2, 0x2fa, 0x2f8, 0x3, 0x2, 0x2, 
    0x2, 0x2fa, 0x2fb, 0x3, 0x2, 0x2, 0x2, 0x2fb, 0x2fd, 0x3, 0x2, 0x2, 
    0x2, 0x2fc, 0x2fa, 0x3, 0x2, 0x2, 0x2, 0x2fd, 0x2fe, 0x7, 0x7b, 0x2, 
    0x2, 0x2fe, 0x77, 0x3, 0x2, 0x2, 0x2, 0x2ff, 0x300, 0x7, 0x16, 0x2, 
    0x2, 0x300, 0x301, 0x7, 0x7a, 0x2, 0x2, 0x301, 0x306, 0x5, 0x80, 0x41, 
    0x2, 0x302, 0x303, 0x7, 0xd, 0x2, 0x2, 0x303, 0x305, 0x5, 0x80, 0x41, 
    0x2, 0x304, 0x302, 0x3, 0x2, 0x2, 0x2, 0x305, 0x308, 0x3, 0x2, 0x2, 
    0x2, 0x306, 0x304, 0x3, 0x2, 0x2, 0x2, 0x306, 0x307, 0x3, 0x2, 0x2, 
    0x2, 0x307, 0x309, 0x3, 0x2, 0x2, 0x2, 0x308, 0x306, 0x3, 0x2, 0x2, 
    0x2, 0x309, 0x30a, 0x7, 0x7b, 0x2, 0x2, 0x30a, 0x79, 0x3, 0x2, 0x2, 
    0x2, 0x30b, 0x30c, 0x7, 0x18, 0x2, 0x2, 0x30c, 0x30d, 0x7, 0x7a, 0x2, 
    0x2, 0x30d, 0x312, 0x5, 0x7e, 0x40, 0x2, 0x30e, 0x30f, 0x7, 0xd, 0x2, 
    0x2, 0x30f, 0x311, 0x5, 0x7e, 0x40, 0x2, 0x310, 0x30e, 0x3, 0x2, 0x2, 
    0x2, 0x311, 0x314, 0x3, 0x2, 0x2, 0x2, 0x312, 0x310, 0x3, 0x2, 0x2, 
    0x2, 0x312, 0x313, 0x3, 0x2, 0x2, 0x2, 0x313, 0x315, 0x3, 0x2, 0x2, 
    0x2, 0x314, 0x312, 0x3, 0x2, 0x2, 0x2, 0x315, 0x316, 0x7, 0x7b, 0x2, 
    0x2, 0x316, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x317, 0x31d, 0x5, 0x82, 0x42, 
    0x2, 0x318, 0x319, 0x7, 0x7a, 0x2, 0x2, 0x319, 0x31a, 0x5, 0x82, 0x42, 
    0x2, 0x31a, 0x31b, 0x7, 0x7b, 0x2, 0x2, 0x31b, 0x31d, 0x3, 0x2, 0x2, 
    0x2, 0x31c, 0x317, 0x3, 0x2, 0x2, 0x2, 0x31c, 0x318, 0x3, 0x2, 0x2, 
    0x2, 0x31d, 0x7d, 0x3, 0x2, 0x2, 0x2, 0x31e, 0x31f, 0x7, 0x7a, 0x2, 
    0x2, 0x31f, 0x324, 0x5, 0x80, 0x41, 0x2, 0x320, 0x321, 0x7, 0xd, 0x2, 
    0x2, 0x321, 0x323, 0x5, 0x80, 0x41, 0x2, 0x322, 0x320, 0x3, 0x2, 0x2, 
    0x2, 0x323, 0x326, 0x3, 0x2, 0x2, 0x2, 0x324, 0x322, 0x3, 0x2, 0x2, 
    0x2, 0x324, 0x325, 0x3, 0x2, 0x2, 0x2, 0x325, 0x327, 0x3, 0x2, 0x2, 
    0x2, 0x326, 0x324, 0x3, 0x2, 0x2, 0x2, 0x327, 0x328, 0x7, 0x7b, 0x2, 
    0x2, 0x328, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x329, 0x32a, 0x7, 0x7a, 0x2, 
    0x2, 0x32a, 0x32f, 0x5, 0x82, 0x42, 0x2, 0x32b, 0x32c, 0x7, 0xd, 0x2, 
    0x2, 0x32c, 0x32e, 0x5, 0x82, 0x42, 0x2, 0x32d, 0x32b, 0x3, 0x2, 0x2, 
    0x2, 0x32e, 0x331, 0x3, 0x2, 0x2, 0x2, 0x32f, 0x32d, 0x3, 0x2, 0x2, 
    0x2, 0x32f, 0x330, 0x3, 0x2, 0x2, 0x2, 0x330, 0x332, 0x3, 0x2, 0x2, 
    0x2, 0x331, 0x32f, 0x3, 0x2, 0x2, 0x2, 0x332, 0x333, 0x7, 0x7b, 0x2, 
    0x2, 0x333, 0x81, 0x3, 0x2, 0x2, 0x2, 0x334, 0x335, 0x9, 0xd, 0x2, 0x2, 
    0x335, 0x336, 0x9, 0xd, 0x2, 0x2, 0x336, 0x83, 0x3, 0x2, 0x2, 0x2, 0x2d, 
    0x87, 0x95, 0x9a, 0xa2, 0xab, 0xb4, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcf, 
    0x100, 0x105, 0x10c, 0x112, 0x12a, 0x12f, 0x133, 0x139, 0x144, 0x149, 
    0x150, 0x158, 0x160, 0x168, 0x172, 0x178, 0x17b, 0x187, 0x194, 0x19b, 
    0x1b1, 0x2af, 0x2db, 0x2dd, 0x2e6, 0x2fa, 0x306, 0x312, 0x31c, 0x324, 
    0x32f, 
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

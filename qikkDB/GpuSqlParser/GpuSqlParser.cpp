
// Generated from C:/Users/AndrejFusekInstarea/Documents/GPU-DB/qikkDB/GpuSqlParser\GpuSqlParser.g4 by ANTLR 4.8


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
    setState(169);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << GpuSqlParser::CREATE)
      | (1ULL << GpuSqlParser::DROP)
      | (1ULL << GpuSqlParser::ALTER)
      | (1ULL << GpuSqlParser::INSERT)
      | (1ULL << GpuSqlParser::SHOW)
      | (1ULL << GpuSqlParser::SELECT))) != 0)) {
      setState(166);
      statement();
      setState(171);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(172);
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

GpuSqlParser::SqlAlterDatabaseContext* GpuSqlParser::StatementContext::sqlAlterDatabase() {
  return getRuleContext<GpuSqlParser::SqlAlterDatabaseContext>(0);
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

GpuSqlParser::ShowQueryTypesContext* GpuSqlParser::StatementContext::showQueryTypes() {
  return getRuleContext<GpuSqlParser::ShowQueryTypesContext>(0);
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
    setState(185);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(174);
      sqlSelect();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(175);
      sqlCreateDb();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(176);
      sqlDropDb();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(177);
      sqlCreateTable();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(178);
      sqlDropTable();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(179);
      sqlAlterTable();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(180);
      sqlAlterDatabase();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(181);
      sqlCreateIndex();
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(182);
      sqlInsertInto();
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(183);
      showStatement();
      break;
    }

    case 11: {
      enterOuterAlt(_localctx, 11);
      setState(184);
      showQueryTypes();
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

GpuSqlParser::ShowConstraintsContext* GpuSqlParser::ShowStatementContext::showConstraints() {
  return getRuleContext<GpuSqlParser::ShowConstraintsContext>(0);
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
    setState(191);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      setState(187);
      showDatabases();
      break;
    }

    case 2: {
      setState(188);
      showTables();
      break;
    }

    case 3: {
      setState(189);
      showColumns();
      break;
    }

    case 4: {
      setState(190);
      showConstraints();
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

//----------------- ShowDatabasesContext ------------------------------------------------------------------

GpuSqlParser::ShowDatabasesContext::ShowDatabasesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ShowDatabasesContext::SHOW() {
  return getToken(GpuSqlParser::SHOW, 0);
}

tree::TerminalNode* GpuSqlParser::ShowDatabasesContext::DATABASES() {
  return getToken(GpuSqlParser::DATABASES, 0);
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
    setState(193);
    match(GpuSqlParser::SHOW);
    setState(194);
    match(GpuSqlParser::DATABASES);
    setState(195);
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

tree::TerminalNode* GpuSqlParser::ShowTablesContext::SHOW() {
  return getToken(GpuSqlParser::SHOW, 0);
}

tree::TerminalNode* GpuSqlParser::ShowTablesContext::TABLES() {
  return getToken(GpuSqlParser::TABLES, 0);
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
    setState(197);
    match(GpuSqlParser::SHOW);
    setState(198);
    match(GpuSqlParser::TABLES);
    setState(201);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(199);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(200);
      database();
    }
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

//----------------- ShowColumnsContext ------------------------------------------------------------------

GpuSqlParser::ShowColumnsContext::ShowColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ShowColumnsContext::SHOW() {
  return getToken(GpuSqlParser::SHOW, 0);
}

tree::TerminalNode* GpuSqlParser::ShowColumnsContext::COLUMNS() {
  return getToken(GpuSqlParser::COLUMNS, 0);
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
    setState(205);
    match(GpuSqlParser::SHOW);
    setState(206);
    match(GpuSqlParser::COLUMNS);
    setState(207);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(208);
    table();
    setState(211);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(209);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(210);
      database();
    }
    setState(213);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShowConstraintsContext ------------------------------------------------------------------

GpuSqlParser::ShowConstraintsContext::ShowConstraintsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ShowConstraintsContext::SHOW() {
  return getToken(GpuSqlParser::SHOW, 0);
}

tree::TerminalNode* GpuSqlParser::ShowConstraintsContext::CONSTRAINTS() {
  return getToken(GpuSqlParser::CONSTRAINTS, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::ShowConstraintsContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}

tree::TerminalNode* GpuSqlParser::ShowConstraintsContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::ShowConstraintsContext::FROM() {
  return getTokens(GpuSqlParser::FROM);
}

tree::TerminalNode* GpuSqlParser::ShowConstraintsContext::FROM(size_t i) {
  return getToken(GpuSqlParser::FROM, i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::ShowConstraintsContext::IN() {
  return getTokens(GpuSqlParser::IN);
}

tree::TerminalNode* GpuSqlParser::ShowConstraintsContext::IN(size_t i) {
  return getToken(GpuSqlParser::IN, i);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::ShowConstraintsContext::database() {
  return getRuleContext<GpuSqlParser::DatabaseContext>(0);
}


size_t GpuSqlParser::ShowConstraintsContext::getRuleIndex() const {
  return GpuSqlParser::RuleShowConstraints;
}

void GpuSqlParser::ShowConstraintsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShowConstraints(this);
}

void GpuSqlParser::ShowConstraintsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShowConstraints(this);
}

GpuSqlParser::ShowConstraintsContext* GpuSqlParser::showConstraints() {
  ShowConstraintsContext *_localctx = _tracker.createInstance<ShowConstraintsContext>(_ctx, getState());
  enterRule(_localctx, 12, GpuSqlParser::RuleShowConstraints);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(215);
    match(GpuSqlParser::SHOW);
    setState(216);
    match(GpuSqlParser::CONSTRAINTS);
    setState(217);
    _la = _input->LA(1);
    if (!(_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(218);
    table();
    setState(221);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FROM

    || _la == GpuSqlParser::IN) {
      setState(219);
      _la = _input->LA(1);
      if (!(_la == GpuSqlParser::FROM

      || _la == GpuSqlParser::IN)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(220);
      database();
    }
    setState(223);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShowQueryTypesContext ------------------------------------------------------------------

GpuSqlParser::ShowQueryTypesContext::ShowQueryTypesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ShowQueryTypesContext::SHOW() {
  return getToken(GpuSqlParser::SHOW, 0);
}

tree::TerminalNode* GpuSqlParser::ShowQueryTypesContext::QUERY() {
  return getToken(GpuSqlParser::QUERY, 0);
}

tree::TerminalNode* GpuSqlParser::ShowQueryTypesContext::COLUMN() {
  return getToken(GpuSqlParser::COLUMN, 0);
}

tree::TerminalNode* GpuSqlParser::ShowQueryTypesContext::TYPES() {
  return getToken(GpuSqlParser::TYPES, 0);
}

GpuSqlParser::SqlSelectContext* GpuSqlParser::ShowQueryTypesContext::sqlSelect() {
  return getRuleContext<GpuSqlParser::SqlSelectContext>(0);
}


size_t GpuSqlParser::ShowQueryTypesContext::getRuleIndex() const {
  return GpuSqlParser::RuleShowQueryTypes;
}

void GpuSqlParser::ShowQueryTypesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterShowQueryTypes(this);
}

void GpuSqlParser::ShowQueryTypesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitShowQueryTypes(this);
}

GpuSqlParser::ShowQueryTypesContext* GpuSqlParser::showQueryTypes() {
  ShowQueryTypesContext *_localctx = _tracker.createInstance<ShowQueryTypesContext>(_ctx, getState());
  enterRule(_localctx, 14, GpuSqlParser::RuleShowQueryTypes);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(225);
    match(GpuSqlParser::SHOW);
    setState(226);
    match(GpuSqlParser::QUERY);
    setState(227);
    match(GpuSqlParser::COLUMN);
    setState(228);
    match(GpuSqlParser::TYPES);
    setState(229);
    sqlSelect();
   
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

tree::TerminalNode* GpuSqlParser::SqlSelectContext::GROUP() {
  return getToken(GpuSqlParser::GROUP, 0);
}

std::vector<tree::TerminalNode *> GpuSqlParser::SqlSelectContext::BY() {
  return getTokens(GpuSqlParser::BY);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::BY(size_t i) {
  return getToken(GpuSqlParser::BY, i);
}

GpuSqlParser::GroupByColumnsContext* GpuSqlParser::SqlSelectContext::groupByColumns() {
  return getRuleContext<GpuSqlParser::GroupByColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlSelectContext::ORDER() {
  return getToken(GpuSqlParser::ORDER, 0);
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
  enterRule(_localctx, 16, GpuSqlParser::RuleSqlSelect);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(231);
    match(GpuSqlParser::SELECT);
    setState(232);
    selectColumns();
    setState(233);
    match(GpuSqlParser::FROM);
    setState(234);
    fromTables();
    setState(236);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FULL

    || _la == GpuSqlParser::INNER || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (GpuSqlParser::JOIN - 64))
      | (1ULL << (GpuSqlParser::LEFT - 64))
      | (1ULL << (GpuSqlParser::RIGHT - 64)))) != 0)) {
      setState(235);
      joinClauses();
    }
    setState(240);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::WHERE) {
      setState(238);
      match(GpuSqlParser::WHERE);
      setState(239);
      whereClause();
    }
    setState(245);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::GROUP) {
      setState(242);
      match(GpuSqlParser::GROUP);
      setState(243);
      match(GpuSqlParser::BY);
      setState(244);
      groupByColumns();
    }
    setState(250);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::ORDER) {
      setState(247);
      match(GpuSqlParser::ORDER);
      setState(248);
      match(GpuSqlParser::BY);
      setState(249);
      orderByColumns();
    }
    setState(254);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::LIMIT) {
      setState(252);
      match(GpuSqlParser::LIMIT);
      setState(253);
      limit();
    }
    setState(258);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::OFFSET) {
      setState(256);
      match(GpuSqlParser::OFFSET);
      setState(257);
      offset();
    }
    setState(260);
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

tree::TerminalNode* GpuSqlParser::SqlCreateDbContext::CREATE() {
  return getToken(GpuSqlParser::CREATE, 0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateDbContext::DATABASE() {
  return getToken(GpuSqlParser::DATABASE, 0);
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
  enterRule(_localctx, 18, GpuSqlParser::RuleSqlCreateDb);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(262);
    match(GpuSqlParser::CREATE);
    setState(263);
    match(GpuSqlParser::DATABASE);
    setState(264);
    database();
    setState(266);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::INTLIT) {
      setState(265);
      blockSize();
    }
    setState(268);
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

tree::TerminalNode* GpuSqlParser::SqlDropDbContext::DROP() {
  return getToken(GpuSqlParser::DROP, 0);
}

tree::TerminalNode* GpuSqlParser::SqlDropDbContext::DATABASE() {
  return getToken(GpuSqlParser::DATABASE, 0);
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
  enterRule(_localctx, 20, GpuSqlParser::RuleSqlDropDb);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(270);
    match(GpuSqlParser::DROP);
    setState(271);
    match(GpuSqlParser::DATABASE);
    setState(272);
    database();
    setState(273);
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

tree::TerminalNode* GpuSqlParser::SqlCreateTableContext::CREATE() {
  return getToken(GpuSqlParser::CREATE, 0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateTableContext::TABLE() {
  return getToken(GpuSqlParser::TABLE, 0);
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

GpuSqlParser::BlockSizeContext* GpuSqlParser::SqlCreateTableContext::blockSize() {
  return getRuleContext<GpuSqlParser::BlockSizeContext>(0);
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
  enterRule(_localctx, 22, GpuSqlParser::RuleSqlCreateTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(275);
    match(GpuSqlParser::CREATE);
    setState(276);
    match(GpuSqlParser::TABLE);
    setState(277);
    table();
    setState(279);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::INTLIT) {
      setState(278);
      blockSize();
    }
    setState(281);
    match(GpuSqlParser::LPAREN);
    setState(282);
    newTableEntries();
    setState(283);
    match(GpuSqlParser::RPAREN);
    setState(284);
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

tree::TerminalNode* GpuSqlParser::SqlDropTableContext::DROP() {
  return getToken(GpuSqlParser::DROP, 0);
}

tree::TerminalNode* GpuSqlParser::SqlDropTableContext::TABLE() {
  return getToken(GpuSqlParser::TABLE, 0);
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
  enterRule(_localctx, 24, GpuSqlParser::RuleSqlDropTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(286);
    match(GpuSqlParser::DROP);
    setState(287);
    match(GpuSqlParser::TABLE);
    setState(288);
    table();
    setState(289);
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

tree::TerminalNode* GpuSqlParser::SqlAlterTableContext::ALTER() {
  return getToken(GpuSqlParser::ALTER, 0);
}

tree::TerminalNode* GpuSqlParser::SqlAlterTableContext::TABLE() {
  return getToken(GpuSqlParser::TABLE, 0);
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
  enterRule(_localctx, 26, GpuSqlParser::RuleSqlAlterTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(291);
    match(GpuSqlParser::ALTER);
    setState(292);
    match(GpuSqlParser::TABLE);
    setState(293);
    table();
    setState(294);
    alterTableEntries();
    setState(295);
    match(GpuSqlParser::SEMICOL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SqlAlterDatabaseContext ------------------------------------------------------------------

GpuSqlParser::SqlAlterDatabaseContext::SqlAlterDatabaseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::SqlAlterDatabaseContext::ALTER() {
  return getToken(GpuSqlParser::ALTER, 0);
}

tree::TerminalNode* GpuSqlParser::SqlAlterDatabaseContext::DATABASE() {
  return getToken(GpuSqlParser::DATABASE, 0);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::SqlAlterDatabaseContext::database() {
  return getRuleContext<GpuSqlParser::DatabaseContext>(0);
}

GpuSqlParser::AlterDatabaseEntriesContext* GpuSqlParser::SqlAlterDatabaseContext::alterDatabaseEntries() {
  return getRuleContext<GpuSqlParser::AlterDatabaseEntriesContext>(0);
}

tree::TerminalNode* GpuSqlParser::SqlAlterDatabaseContext::SEMICOL() {
  return getToken(GpuSqlParser::SEMICOL, 0);
}


size_t GpuSqlParser::SqlAlterDatabaseContext::getRuleIndex() const {
  return GpuSqlParser::RuleSqlAlterDatabase;
}

void GpuSqlParser::SqlAlterDatabaseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSqlAlterDatabase(this);
}

void GpuSqlParser::SqlAlterDatabaseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSqlAlterDatabase(this);
}

GpuSqlParser::SqlAlterDatabaseContext* GpuSqlParser::sqlAlterDatabase() {
  SqlAlterDatabaseContext *_localctx = _tracker.createInstance<SqlAlterDatabaseContext>(_ctx, getState());
  enterRule(_localctx, 28, GpuSqlParser::RuleSqlAlterDatabase);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(297);
    match(GpuSqlParser::ALTER);
    setState(298);
    match(GpuSqlParser::DATABASE);
    setState(299);
    database();
    setState(300);
    alterDatabaseEntries();
    setState(301);
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

tree::TerminalNode* GpuSqlParser::SqlCreateIndexContext::CREATE() {
  return getToken(GpuSqlParser::CREATE, 0);
}

tree::TerminalNode* GpuSqlParser::SqlCreateIndexContext::INDEX() {
  return getToken(GpuSqlParser::INDEX, 0);
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
  enterRule(_localctx, 30, GpuSqlParser::RuleSqlCreateIndex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(303);
    match(GpuSqlParser::CREATE);
    setState(304);
    match(GpuSqlParser::INDEX);
    setState(305);
    indexName();
    setState(306);
    match(GpuSqlParser::ON);
    setState(307);
    table();
    setState(308);
    match(GpuSqlParser::LPAREN);
    setState(309);
    indexColumns();
    setState(310);
    match(GpuSqlParser::RPAREN);
    setState(311);
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

tree::TerminalNode* GpuSqlParser::SqlInsertIntoContext::INSERT() {
  return getToken(GpuSqlParser::INSERT, 0);
}

tree::TerminalNode* GpuSqlParser::SqlInsertIntoContext::INTO() {
  return getToken(GpuSqlParser::INTO, 0);
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
  enterRule(_localctx, 32, GpuSqlParser::RuleSqlInsertInto);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(313);
    match(GpuSqlParser::INSERT);
    setState(314);
    match(GpuSqlParser::INTO);
    setState(315);
    table();
    setState(316);
    match(GpuSqlParser::LPAREN);
    setState(317);
    insertIntoColumns();
    setState(318);
    match(GpuSqlParser::RPAREN);
    setState(319);
    match(GpuSqlParser::VALUES);
    setState(320);
    match(GpuSqlParser::LPAREN);
    setState(321);
    insertIntoValues();
    setState(322);
    match(GpuSqlParser::RPAREN);
    setState(323);
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
  enterRule(_localctx, 34, GpuSqlParser::RuleNewTableEntries);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(325);
    newTableEntry();
    setState(330);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(326);
      match(GpuSqlParser::COMMA);
      setState(327);
      newTableEntry();
      setState(332);
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

GpuSqlParser::NewTableConstraintContext* GpuSqlParser::NewTableEntryContext::newTableConstraint() {
  return getRuleContext<GpuSqlParser::NewTableConstraintContext>(0);
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
  enterRule(_localctx, 36, GpuSqlParser::RuleNewTableEntry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(335);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::DELIMID:
      case GpuSqlParser::ID: {
        setState(333);
        newTableColumn();
        break;
      }

      case GpuSqlParser::INDEX:
      case GpuSqlParser::UNIQUE:
      case GpuSqlParser::NOT: {
        setState(334);
        newTableConstraint();
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

//----------------- AlterDatabaseEntriesContext ------------------------------------------------------------------

GpuSqlParser::AlterDatabaseEntriesContext::AlterDatabaseEntriesContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::AlterDatabaseEntryContext *> GpuSqlParser::AlterDatabaseEntriesContext::alterDatabaseEntry() {
  return getRuleContexts<GpuSqlParser::AlterDatabaseEntryContext>();
}

GpuSqlParser::AlterDatabaseEntryContext* GpuSqlParser::AlterDatabaseEntriesContext::alterDatabaseEntry(size_t i) {
  return getRuleContext<GpuSqlParser::AlterDatabaseEntryContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::AlterDatabaseEntriesContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::AlterDatabaseEntriesContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::AlterDatabaseEntriesContext::getRuleIndex() const {
  return GpuSqlParser::RuleAlterDatabaseEntries;
}

void GpuSqlParser::AlterDatabaseEntriesContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAlterDatabaseEntries(this);
}

void GpuSqlParser::AlterDatabaseEntriesContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAlterDatabaseEntries(this);
}

GpuSqlParser::AlterDatabaseEntriesContext* GpuSqlParser::alterDatabaseEntries() {
  AlterDatabaseEntriesContext *_localctx = _tracker.createInstance<AlterDatabaseEntriesContext>(_ctx, getState());
  enterRule(_localctx, 38, GpuSqlParser::RuleAlterDatabaseEntries);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(337);
    alterDatabaseEntry();
    setState(342);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(338);
      match(GpuSqlParser::COMMA);
      setState(339);
      alterDatabaseEntry();
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

//----------------- AlterDatabaseEntryContext ------------------------------------------------------------------

GpuSqlParser::AlterDatabaseEntryContext::AlterDatabaseEntryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::RenameDatabaseContext* GpuSqlParser::AlterDatabaseEntryContext::renameDatabase() {
  return getRuleContext<GpuSqlParser::RenameDatabaseContext>(0);
}

GpuSqlParser::AlterBlockSizeContext* GpuSqlParser::AlterDatabaseEntryContext::alterBlockSize() {
  return getRuleContext<GpuSqlParser::AlterBlockSizeContext>(0);
}


size_t GpuSqlParser::AlterDatabaseEntryContext::getRuleIndex() const {
  return GpuSqlParser::RuleAlterDatabaseEntry;
}

void GpuSqlParser::AlterDatabaseEntryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAlterDatabaseEntry(this);
}

void GpuSqlParser::AlterDatabaseEntryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAlterDatabaseEntry(this);
}

GpuSqlParser::AlterDatabaseEntryContext* GpuSqlParser::alterDatabaseEntry() {
  AlterDatabaseEntryContext *_localctx = _tracker.createInstance<AlterDatabaseEntryContext>(_ctx, getState());
  enterRule(_localctx, 40, GpuSqlParser::RuleAlterDatabaseEntry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(347);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::RENAME: {
        setState(345);
        renameDatabase();
        break;
      }

      case GpuSqlParser::ALTER: {
        setState(346);
        alterBlockSize();
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

//----------------- RenameDatabaseContext ------------------------------------------------------------------

GpuSqlParser::RenameDatabaseContext::RenameDatabaseContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::RenameDatabaseContext::RENAME() {
  return getToken(GpuSqlParser::RENAME, 0);
}

tree::TerminalNode* GpuSqlParser::RenameDatabaseContext::TO() {
  return getToken(GpuSqlParser::TO, 0);
}

GpuSqlParser::DatabaseContext* GpuSqlParser::RenameDatabaseContext::database() {
  return getRuleContext<GpuSqlParser::DatabaseContext>(0);
}


size_t GpuSqlParser::RenameDatabaseContext::getRuleIndex() const {
  return GpuSqlParser::RuleRenameDatabase;
}

void GpuSqlParser::RenameDatabaseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRenameDatabase(this);
}

void GpuSqlParser::RenameDatabaseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRenameDatabase(this);
}

GpuSqlParser::RenameDatabaseContext* GpuSqlParser::renameDatabase() {
  RenameDatabaseContext *_localctx = _tracker.createInstance<RenameDatabaseContext>(_ctx, getState());
  enterRule(_localctx, 42, GpuSqlParser::RuleRenameDatabase);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(349);
    match(GpuSqlParser::RENAME);
    setState(350);
    match(GpuSqlParser::TO);
    setState(351);
    database();
   
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
  enterRule(_localctx, 44, GpuSqlParser::RuleAlterTableEntries);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(353);
    alterTableEntry();
    setState(358);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(354);
      match(GpuSqlParser::COMMA);
      setState(355);
      alterTableEntry();
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

GpuSqlParser::RenameColumnContext* GpuSqlParser::AlterTableEntryContext::renameColumn() {
  return getRuleContext<GpuSqlParser::RenameColumnContext>(0);
}

GpuSqlParser::RenameTableContext* GpuSqlParser::AlterTableEntryContext::renameTable() {
  return getRuleContext<GpuSqlParser::RenameTableContext>(0);
}

GpuSqlParser::AddConstraintContext* GpuSqlParser::AlterTableEntryContext::addConstraint() {
  return getRuleContext<GpuSqlParser::AddConstraintContext>(0);
}

GpuSqlParser::DropConstraintContext* GpuSqlParser::AlterTableEntryContext::dropConstraint() {
  return getRuleContext<GpuSqlParser::DropConstraintContext>(0);
}

GpuSqlParser::AlterBlockSizeContext* GpuSqlParser::AlterTableEntryContext::alterBlockSize() {
  return getRuleContext<GpuSqlParser::AlterBlockSizeContext>(0);
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
  enterRule(_localctx, 46, GpuSqlParser::RuleAlterTableEntry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(369);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
    case 1: {
      setState(361);
      addColumn();
      break;
    }

    case 2: {
      setState(362);
      dropColumn();
      break;
    }

    case 3: {
      setState(363);
      alterColumn();
      break;
    }

    case 4: {
      setState(364);
      renameColumn();
      break;
    }

    case 5: {
      setState(365);
      renameTable();
      break;
    }

    case 6: {
      setState(366);
      addConstraint();
      break;
    }

    case 7: {
      setState(367);
      dropConstraint();
      break;
    }

    case 8: {
      setState(368);
      alterBlockSize();
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

GpuSqlParser::DatatypeContext* GpuSqlParser::AddColumnContext::datatype() {
  return getRuleContext<GpuSqlParser::DatatypeContext>(0);
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
  enterRule(_localctx, 48, GpuSqlParser::RuleAddColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(371);
    match(GpuSqlParser::ADD);
    setState(372);
    column();
    setState(373);
    datatype();
   
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

tree::TerminalNode* GpuSqlParser::DropColumnContext::DROP() {
  return getToken(GpuSqlParser::DROP, 0);
}

tree::TerminalNode* GpuSqlParser::DropColumnContext::COLUMN() {
  return getToken(GpuSqlParser::COLUMN, 0);
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
  enterRule(_localctx, 50, GpuSqlParser::RuleDropColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(375);
    match(GpuSqlParser::DROP);
    setState(376);
    match(GpuSqlParser::COLUMN);
    setState(377);
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

tree::TerminalNode* GpuSqlParser::AlterColumnContext::ALTER() {
  return getToken(GpuSqlParser::ALTER, 0);
}

tree::TerminalNode* GpuSqlParser::AlterColumnContext::COLUMN() {
  return getToken(GpuSqlParser::COLUMN, 0);
}

GpuSqlParser::ColumnContext* GpuSqlParser::AlterColumnContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
}

GpuSqlParser::DatatypeContext* GpuSqlParser::AlterColumnContext::datatype() {
  return getRuleContext<GpuSqlParser::DatatypeContext>(0);
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
  enterRule(_localctx, 52, GpuSqlParser::RuleAlterColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(379);
    match(GpuSqlParser::ALTER);
    setState(380);
    match(GpuSqlParser::COLUMN);
    setState(381);
    column();
    setState(382);
    datatype();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RenameColumnContext ------------------------------------------------------------------

GpuSqlParser::RenameColumnContext::RenameColumnContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::RenameColumnContext::RENAME() {
  return getToken(GpuSqlParser::RENAME, 0);
}

tree::TerminalNode* GpuSqlParser::RenameColumnContext::COLUMN() {
  return getToken(GpuSqlParser::COLUMN, 0);
}

GpuSqlParser::RenameColumnFromContext* GpuSqlParser::RenameColumnContext::renameColumnFrom() {
  return getRuleContext<GpuSqlParser::RenameColumnFromContext>(0);
}

tree::TerminalNode* GpuSqlParser::RenameColumnContext::TO() {
  return getToken(GpuSqlParser::TO, 0);
}

GpuSqlParser::RenameColumnToContext* GpuSqlParser::RenameColumnContext::renameColumnTo() {
  return getRuleContext<GpuSqlParser::RenameColumnToContext>(0);
}


size_t GpuSqlParser::RenameColumnContext::getRuleIndex() const {
  return GpuSqlParser::RuleRenameColumn;
}

void GpuSqlParser::RenameColumnContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRenameColumn(this);
}

void GpuSqlParser::RenameColumnContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRenameColumn(this);
}

GpuSqlParser::RenameColumnContext* GpuSqlParser::renameColumn() {
  RenameColumnContext *_localctx = _tracker.createInstance<RenameColumnContext>(_ctx, getState());
  enterRule(_localctx, 54, GpuSqlParser::RuleRenameColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(384);
    match(GpuSqlParser::RENAME);
    setState(385);
    match(GpuSqlParser::COLUMN);
    setState(386);
    renameColumnFrom();
    setState(387);
    match(GpuSqlParser::TO);
    setState(388);
    renameColumnTo();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RenameTableContext ------------------------------------------------------------------

GpuSqlParser::RenameTableContext::RenameTableContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::RenameTableContext::RENAME() {
  return getToken(GpuSqlParser::RENAME, 0);
}

tree::TerminalNode* GpuSqlParser::RenameTableContext::TO() {
  return getToken(GpuSqlParser::TO, 0);
}

GpuSqlParser::TableContext* GpuSqlParser::RenameTableContext::table() {
  return getRuleContext<GpuSqlParser::TableContext>(0);
}


size_t GpuSqlParser::RenameTableContext::getRuleIndex() const {
  return GpuSqlParser::RuleRenameTable;
}

void GpuSqlParser::RenameTableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRenameTable(this);
}

void GpuSqlParser::RenameTableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRenameTable(this);
}

GpuSqlParser::RenameTableContext* GpuSqlParser::renameTable() {
  RenameTableContext *_localctx = _tracker.createInstance<RenameTableContext>(_ctx, getState());
  enterRule(_localctx, 56, GpuSqlParser::RuleRenameTable);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(390);
    match(GpuSqlParser::RENAME);
    setState(391);
    match(GpuSqlParser::TO);
    setState(392);
    table();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AddConstraintContext ------------------------------------------------------------------

GpuSqlParser::AddConstraintContext::AddConstraintContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::AddConstraintContext::ADD() {
  return getToken(GpuSqlParser::ADD, 0);
}

GpuSqlParser::ConstraintContext* GpuSqlParser::AddConstraintContext::constraint() {
  return getRuleContext<GpuSqlParser::ConstraintContext>(0);
}

GpuSqlParser::ConstraintNameContext* GpuSqlParser::AddConstraintContext::constraintName() {
  return getRuleContext<GpuSqlParser::ConstraintNameContext>(0);
}

tree::TerminalNode* GpuSqlParser::AddConstraintContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::ConstraintColumnsContext* GpuSqlParser::AddConstraintContext::constraintColumns() {
  return getRuleContext<GpuSqlParser::ConstraintColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::AddConstraintContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}


size_t GpuSqlParser::AddConstraintContext::getRuleIndex() const {
  return GpuSqlParser::RuleAddConstraint;
}

void GpuSqlParser::AddConstraintContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAddConstraint(this);
}

void GpuSqlParser::AddConstraintContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAddConstraint(this);
}

GpuSqlParser::AddConstraintContext* GpuSqlParser::addConstraint() {
  AddConstraintContext *_localctx = _tracker.createInstance<AddConstraintContext>(_ctx, getState());
  enterRule(_localctx, 58, GpuSqlParser::RuleAddConstraint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(394);
    match(GpuSqlParser::ADD);
    setState(395);
    constraint();
    setState(396);
    constraintName();
    setState(397);
    match(GpuSqlParser::LPAREN);
    setState(398);
    constraintColumns();
    setState(399);
    match(GpuSqlParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DropConstraintContext ------------------------------------------------------------------

GpuSqlParser::DropConstraintContext::DropConstraintContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::DropConstraintContext::DROP() {
  return getToken(GpuSqlParser::DROP, 0);
}

GpuSqlParser::ConstraintContext* GpuSqlParser::DropConstraintContext::constraint() {
  return getRuleContext<GpuSqlParser::ConstraintContext>(0);
}

GpuSqlParser::ConstraintNameContext* GpuSqlParser::DropConstraintContext::constraintName() {
  return getRuleContext<GpuSqlParser::ConstraintNameContext>(0);
}


size_t GpuSqlParser::DropConstraintContext::getRuleIndex() const {
  return GpuSqlParser::RuleDropConstraint;
}

void GpuSqlParser::DropConstraintContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDropConstraint(this);
}

void GpuSqlParser::DropConstraintContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDropConstraint(this);
}

GpuSqlParser::DropConstraintContext* GpuSqlParser::dropConstraint() {
  DropConstraintContext *_localctx = _tracker.createInstance<DropConstraintContext>(_ctx, getState());
  enterRule(_localctx, 60, GpuSqlParser::RuleDropConstraint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(401);
    match(GpuSqlParser::DROP);
    setState(402);
    constraint();
    setState(403);
    constraintName();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AlterBlockSizeContext ------------------------------------------------------------------

GpuSqlParser::AlterBlockSizeContext::AlterBlockSizeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::AlterBlockSizeContext::ALTER() {
  return getToken(GpuSqlParser::ALTER, 0);
}

tree::TerminalNode* GpuSqlParser::AlterBlockSizeContext::BLOCK() {
  return getToken(GpuSqlParser::BLOCK, 0);
}

tree::TerminalNode* GpuSqlParser::AlterBlockSizeContext::SIZE() {
  return getToken(GpuSqlParser::SIZE, 0);
}

GpuSqlParser::BlockSizeContext* GpuSqlParser::AlterBlockSizeContext::blockSize() {
  return getRuleContext<GpuSqlParser::BlockSizeContext>(0);
}


size_t GpuSqlParser::AlterBlockSizeContext::getRuleIndex() const {
  return GpuSqlParser::RuleAlterBlockSize;
}

void GpuSqlParser::AlterBlockSizeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAlterBlockSize(this);
}

void GpuSqlParser::AlterBlockSizeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAlterBlockSize(this);
}

GpuSqlParser::AlterBlockSizeContext* GpuSqlParser::alterBlockSize() {
  AlterBlockSizeContext *_localctx = _tracker.createInstance<AlterBlockSizeContext>(_ctx, getState());
  enterRule(_localctx, 62, GpuSqlParser::RuleAlterBlockSize);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(405);
    match(GpuSqlParser::ALTER);
    setState(406);
    match(GpuSqlParser::BLOCK);
    setState(407);
    match(GpuSqlParser::SIZE);
    setState(408);
    blockSize();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RenameColumnFromContext ------------------------------------------------------------------

GpuSqlParser::RenameColumnFromContext::RenameColumnFromContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ColumnContext* GpuSqlParser::RenameColumnFromContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
}


size_t GpuSqlParser::RenameColumnFromContext::getRuleIndex() const {
  return GpuSqlParser::RuleRenameColumnFrom;
}

void GpuSqlParser::RenameColumnFromContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRenameColumnFrom(this);
}

void GpuSqlParser::RenameColumnFromContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRenameColumnFrom(this);
}

GpuSqlParser::RenameColumnFromContext* GpuSqlParser::renameColumnFrom() {
  RenameColumnFromContext *_localctx = _tracker.createInstance<RenameColumnFromContext>(_ctx, getState());
  enterRule(_localctx, 64, GpuSqlParser::RuleRenameColumnFrom);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(410);
    column();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RenameColumnToContext ------------------------------------------------------------------

GpuSqlParser::RenameColumnToContext::RenameColumnToContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ColumnContext* GpuSqlParser::RenameColumnToContext::column() {
  return getRuleContext<GpuSqlParser::ColumnContext>(0);
}


size_t GpuSqlParser::RenameColumnToContext::getRuleIndex() const {
  return GpuSqlParser::RuleRenameColumnTo;
}

void GpuSqlParser::RenameColumnToContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRenameColumnTo(this);
}

void GpuSqlParser::RenameColumnToContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRenameColumnTo(this);
}

GpuSqlParser::RenameColumnToContext* GpuSqlParser::renameColumnTo() {
  RenameColumnToContext *_localctx = _tracker.createInstance<RenameColumnToContext>(_ctx, getState());
  enterRule(_localctx, 66, GpuSqlParser::RuleRenameColumnTo);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(412);
    column();
   
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

GpuSqlParser::DatatypeContext* GpuSqlParser::NewTableColumnContext::datatype() {
  return getRuleContext<GpuSqlParser::DatatypeContext>(0);
}

GpuSqlParser::ConstraintContext* GpuSqlParser::NewTableColumnContext::constraint() {
  return getRuleContext<GpuSqlParser::ConstraintContext>(0);
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
  enterRule(_localctx, 68, GpuSqlParser::RuleNewTableColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(414);
    column();
    setState(415);
    datatype();
    setState(417);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (((((_la - 36) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 36)) & ((1ULL << (GpuSqlParser::INDEX - 36))
      | (1ULL << (GpuSqlParser::UNIQUE - 36))
      | (1ULL << (GpuSqlParser::NOT - 36)))) != 0)) {
      setState(416);
      constraint();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NewTableConstraintContext ------------------------------------------------------------------

GpuSqlParser::NewTableConstraintContext::NewTableConstraintContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::ConstraintContext* GpuSqlParser::NewTableConstraintContext::constraint() {
  return getRuleContext<GpuSqlParser::ConstraintContext>(0);
}

GpuSqlParser::ConstraintNameContext* GpuSqlParser::NewTableConstraintContext::constraintName() {
  return getRuleContext<GpuSqlParser::ConstraintNameContext>(0);
}

tree::TerminalNode* GpuSqlParser::NewTableConstraintContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

GpuSqlParser::ConstraintColumnsContext* GpuSqlParser::NewTableConstraintContext::constraintColumns() {
  return getRuleContext<GpuSqlParser::ConstraintColumnsContext>(0);
}

tree::TerminalNode* GpuSqlParser::NewTableConstraintContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}


size_t GpuSqlParser::NewTableConstraintContext::getRuleIndex() const {
  return GpuSqlParser::RuleNewTableConstraint;
}

void GpuSqlParser::NewTableConstraintContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNewTableConstraint(this);
}

void GpuSqlParser::NewTableConstraintContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNewTableConstraint(this);
}

GpuSqlParser::NewTableConstraintContext* GpuSqlParser::newTableConstraint() {
  NewTableConstraintContext *_localctx = _tracker.createInstance<NewTableConstraintContext>(_ctx, getState());
  enterRule(_localctx, 70, GpuSqlParser::RuleNewTableConstraint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(419);
    constraint();
    setState(420);
    constraintName();
    setState(421);
    match(GpuSqlParser::LPAREN);
    setState(422);
    constraintColumns();
    setState(423);
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
  enterRule(_localctx, 72, GpuSqlParser::RuleSelectColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(427);
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
      case GpuSqlParser::DATETYPE:
      case GpuSqlParser::AVG_AGG:
      case GpuSqlParser::SUM_AGG:
      case GpuSqlParser::MIN_AGG:
      case GpuSqlParser::MAX_AGG:
      case GpuSqlParser::COUNT_AGG:
      case GpuSqlParser::YEAR:
      case GpuSqlParser::MONTH:
      case GpuSqlParser::DAY:
      case GpuSqlParser::HOUR:
      case GpuSqlParser::MINUTE:
      case GpuSqlParser::SECOND:
      case GpuSqlParser::WEEKDAY:
      case GpuSqlParser::DAYOFWEEK:
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
      case GpuSqlParser::GEO_LONGITUDE_TO_TILE_X:
      case GpuSqlParser::GEO_LATITUDE_TO_TILE_Y:
      case GpuSqlParser::GEO_TILE_X_TO_LONGITUDE:
      case GpuSqlParser::GEO_TILE_Y_TO_LATITUDE:
      case GpuSqlParser::MINUS:
      case GpuSqlParser::LPAREN:
      case GpuSqlParser::LOGICAL_NOT:
      case GpuSqlParser::BOOLEANLIT:
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT:
      case GpuSqlParser::ID: {
        setState(425);
        selectColumn();
        break;
      }

      case GpuSqlParser::ASTERISK: {
        setState(426);
        selectAllColumns();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    setState(436);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(429);
      match(GpuSqlParser::COMMA);
      setState(432);
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
        case GpuSqlParser::DATETYPE:
        case GpuSqlParser::AVG_AGG:
        case GpuSqlParser::SUM_AGG:
        case GpuSqlParser::MIN_AGG:
        case GpuSqlParser::MAX_AGG:
        case GpuSqlParser::COUNT_AGG:
        case GpuSqlParser::YEAR:
        case GpuSqlParser::MONTH:
        case GpuSqlParser::DAY:
        case GpuSqlParser::HOUR:
        case GpuSqlParser::MINUTE:
        case GpuSqlParser::SECOND:
        case GpuSqlParser::WEEKDAY:
        case GpuSqlParser::DAYOFWEEK:
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
        case GpuSqlParser::GEO_LONGITUDE_TO_TILE_X:
        case GpuSqlParser::GEO_LATITUDE_TO_TILE_Y:
        case GpuSqlParser::GEO_TILE_X_TO_LONGITUDE:
        case GpuSqlParser::GEO_TILE_Y_TO_LATITUDE:
        case GpuSqlParser::MINUS:
        case GpuSqlParser::LPAREN:
        case GpuSqlParser::LOGICAL_NOT:
        case GpuSqlParser::BOOLEANLIT:
        case GpuSqlParser::FLOATLIT:
        case GpuSqlParser::INTLIT:
        case GpuSqlParser::ID: {
          setState(430);
          selectColumn();
          break;
        }

        case GpuSqlParser::ASTERISK: {
          setState(431);
          selectAllColumns();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(438);
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

tree::TerminalNode* GpuSqlParser::SelectColumnContext::RETPAYLOAD() {
  return getToken(GpuSqlParser::RETPAYLOAD, 0);
}

GpuSqlParser::RetpayloadContext* GpuSqlParser::SelectColumnContext::retpayload() {
  return getRuleContext<GpuSqlParser::RetpayloadContext>(0);
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
  enterRule(_localctx, 74, GpuSqlParser::RuleSelectColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(439);
    expression(0);
    setState(442);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(440);
      match(GpuSqlParser::AS);
      setState(441);
      alias();
    }
    setState(446);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::RETPAYLOAD) {
      setState(444);
      match(GpuSqlParser::RETPAYLOAD);
      setState(445);
      retpayload();
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
  enterRule(_localctx, 76, GpuSqlParser::RuleSelectAllColumns);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(448);
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
  enterRule(_localctx, 78, GpuSqlParser::RuleWhereClause);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(450);
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
  enterRule(_localctx, 80, GpuSqlParser::RuleOrderByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(452);
    orderByColumn();
    setState(457);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(453);
      match(GpuSqlParser::COMMA);
      setState(454);
      orderByColumn();
      setState(459);
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
  enterRule(_localctx, 82, GpuSqlParser::RuleOrderByColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(460);
    expression(0);
    setState(462);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::DIR) {
      setState(461);
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
  enterRule(_localctx, 84, GpuSqlParser::RuleInsertIntoValues);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(464);
    columnValue();
    setState(469);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(465);
      match(GpuSqlParser::COMMA);
      setState(466);
      columnValue();
      setState(471);
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
  enterRule(_localctx, 86, GpuSqlParser::RuleInsertIntoColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(472);
    columnId();
    setState(477);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(473);
      match(GpuSqlParser::COMMA);
      setState(474);
      columnId();
      setState(479);
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
  enterRule(_localctx, 88, GpuSqlParser::RuleIndexColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(480);
    column();
    setState(485);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(481);
      match(GpuSqlParser::COMMA);
      setState(482);
      column();
      setState(487);
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

//----------------- ConstraintColumnsContext ------------------------------------------------------------------

GpuSqlParser::ConstraintColumnsContext::ConstraintColumnsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<GpuSqlParser::ColumnContext *> GpuSqlParser::ConstraintColumnsContext::column() {
  return getRuleContexts<GpuSqlParser::ColumnContext>();
}

GpuSqlParser::ColumnContext* GpuSqlParser::ConstraintColumnsContext::column(size_t i) {
  return getRuleContext<GpuSqlParser::ColumnContext>(i);
}

std::vector<tree::TerminalNode *> GpuSqlParser::ConstraintColumnsContext::COMMA() {
  return getTokens(GpuSqlParser::COMMA);
}

tree::TerminalNode* GpuSqlParser::ConstraintColumnsContext::COMMA(size_t i) {
  return getToken(GpuSqlParser::COMMA, i);
}


size_t GpuSqlParser::ConstraintColumnsContext::getRuleIndex() const {
  return GpuSqlParser::RuleConstraintColumns;
}

void GpuSqlParser::ConstraintColumnsContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstraintColumns(this);
}

void GpuSqlParser::ConstraintColumnsContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstraintColumns(this);
}

GpuSqlParser::ConstraintColumnsContext* GpuSqlParser::constraintColumns() {
  ConstraintColumnsContext *_localctx = _tracker.createInstance<ConstraintColumnsContext>(_ctx, getState());
  enterRule(_localctx, 90, GpuSqlParser::RuleConstraintColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(488);
    column();
    setState(493);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(489);
      match(GpuSqlParser::COMMA);
      setState(490);
      column();
      setState(495);
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
  enterRule(_localctx, 92, GpuSqlParser::RuleGroupByColumns);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(496);
    groupByColumn();
    setState(501);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(497);
      match(GpuSqlParser::COMMA);
      setState(498);
      groupByColumn();
      setState(503);
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
  enterRule(_localctx, 94, GpuSqlParser::RuleGroupByColumn);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(504);
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
  enterRule(_localctx, 96, GpuSqlParser::RuleFromTables);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(506);
    fromTable();
    setState(511);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(507);
      match(GpuSqlParser::COMMA);
      setState(508);
      fromTable();
      setState(513);
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
  enterRule(_localctx, 98, GpuSqlParser::RuleJoinClauses);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(515); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(514);
      joinClause();
      setState(517); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == GpuSqlParser::FULL

    || _la == GpuSqlParser::INNER || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (GpuSqlParser::JOIN - 64))
      | (1ULL << (GpuSqlParser::LEFT - 64))
      | (1ULL << (GpuSqlParser::RIGHT - 64)))) != 0));
   
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
  enterRule(_localctx, 100, GpuSqlParser::RuleJoinClause);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(520);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::FULL

    || _la == GpuSqlParser::INNER || _la == GpuSqlParser::LEFT

    || _la == GpuSqlParser::RIGHT) {
      setState(519);
      joinType();
    }
    setState(522);
    match(GpuSqlParser::JOIN);
    setState(523);
    joinTable();
    setState(524);
    match(GpuSqlParser::ON);
    setState(525);
    joinColumnLeft();
    setState(526);
    joinOperator();
    setState(527);
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
  enterRule(_localctx, 102, GpuSqlParser::RuleJoinTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(529);
    table();
    setState(532);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(530);
      match(GpuSqlParser::AS);
      setState(531);
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
  enterRule(_localctx, 104, GpuSqlParser::RuleJoinColumnLeft);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(534);
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
  enterRule(_localctx, 106, GpuSqlParser::RuleJoinColumnRight);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(536);
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
  enterRule(_localctx, 108, GpuSqlParser::RuleJoinOperator);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(538);
    _la = _input->LA(1);
    if (!(((((_la - 140) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 140)) & ((1ULL << (GpuSqlParser::EQUALS - 140))
      | (1ULL << (GpuSqlParser::NOTEQUALS - 140))
      | (1ULL << (GpuSqlParser::NOTEQUALS_GT_LT - 140))
      | (1ULL << (GpuSqlParser::GREATER - 140))
      | (1ULL << (GpuSqlParser::LESS - 140))
      | (1ULL << (GpuSqlParser::GREATEREQ - 140))
      | (1ULL << (GpuSqlParser::LESSEQ - 140)))) != 0))) {
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

tree::TerminalNode* GpuSqlParser::JoinTypeContext::FULL() {
  return getToken(GpuSqlParser::FULL, 0);
}

tree::TerminalNode* GpuSqlParser::JoinTypeContext::OUTER() {
  return getToken(GpuSqlParser::OUTER, 0);
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
  enterRule(_localctx, 110, GpuSqlParser::RuleJoinType);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(545);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::INNER: {
        setState(540);
        match(GpuSqlParser::INNER);
        break;
      }

      case GpuSqlParser::LEFT: {
        setState(541);
        match(GpuSqlParser::LEFT);
        break;
      }

      case GpuSqlParser::RIGHT: {
        setState(542);
        match(GpuSqlParser::RIGHT);
        break;
      }

      case GpuSqlParser::FULL: {
        setState(543);
        match(GpuSqlParser::FULL);
        setState(544);
        match(GpuSqlParser::OUTER);
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
  enterRule(_localctx, 112, GpuSqlParser::RuleFromTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(547);
    table();
    setState(550);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == GpuSqlParser::AS) {
      setState(548);
      match(GpuSqlParser::AS);
      setState(549);
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
  enterRule(_localctx, 114, GpuSqlParser::RuleColumnId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(557);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 39, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(552);
      column();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(553);
      table();
      setState(554);
      match(GpuSqlParser::DOT);
      setState(555);
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
  enterRule(_localctx, 116, GpuSqlParser::RuleTable);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(559);
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
  enterRule(_localctx, 118, GpuSqlParser::RuleColumn);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(561);
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
  enterRule(_localctx, 120, GpuSqlParser::RuleDatabase);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(563);
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
  enterRule(_localctx, 122, GpuSqlParser::RuleAlias);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(565);
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
  enterRule(_localctx, 124, GpuSqlParser::RuleIndexName);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(567);
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

//----------------- ConstraintNameContext ------------------------------------------------------------------

GpuSqlParser::ConstraintNameContext::ConstraintNameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ConstraintNameContext::ID() {
  return getToken(GpuSqlParser::ID, 0);
}

tree::TerminalNode* GpuSqlParser::ConstraintNameContext::DELIMID() {
  return getToken(GpuSqlParser::DELIMID, 0);
}


size_t GpuSqlParser::ConstraintNameContext::getRuleIndex() const {
  return GpuSqlParser::RuleConstraintName;
}

void GpuSqlParser::ConstraintNameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstraintName(this);
}

void GpuSqlParser::ConstraintNameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstraintName(this);
}

GpuSqlParser::ConstraintNameContext* GpuSqlParser::constraintName() {
  ConstraintNameContext *_localctx = _tracker.createInstance<ConstraintNameContext>(_ctx, getState());
  enterRule(_localctx, 126, GpuSqlParser::RuleConstraintName);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(569);
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
  enterRule(_localctx, 128, GpuSqlParser::RuleLimit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(571);
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
  enterRule(_localctx, 130, GpuSqlParser::RuleOffset);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(573);
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
  enterRule(_localctx, 132, GpuSqlParser::RuleBlockSize);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(575);
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

tree::TerminalNode* GpuSqlParser::ColumnValueContext::NULL_T() {
  return getToken(GpuSqlParser::NULL_T, 0);
}

tree::TerminalNode* GpuSqlParser::ColumnValueContext::STRING() {
  return getToken(GpuSqlParser::STRING, 0);
}

tree::TerminalNode* GpuSqlParser::ColumnValueContext::DATETIMELIT() {
  return getToken(GpuSqlParser::DATETIMELIT, 0);
}

tree::TerminalNode* GpuSqlParser::ColumnValueContext::BOOLEANLIT() {
  return getToken(GpuSqlParser::BOOLEANLIT, 0);
}

tree::TerminalNode* GpuSqlParser::ColumnValueContext::MINUS() {
  return getToken(GpuSqlParser::MINUS, 0);
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
  enterRule(_localctx, 134, GpuSqlParser::RuleColumnValue);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(590);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 42, _ctx)) {
    case 1: {
      setState(578);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == GpuSqlParser::MINUS) {
        setState(577);
        match(GpuSqlParser::MINUS);
      }
      setState(580);
      match(GpuSqlParser::INTLIT);
      break;
    }

    case 2: {
      setState(582);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == GpuSqlParser::MINUS) {
        setState(581);
        match(GpuSqlParser::MINUS);
      }
      setState(584);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 3: {
      setState(585);
      geometry();
      break;
    }

    case 4: {
      setState(586);
      match(GpuSqlParser::NULL_T);
      break;
    }

    case 5: {
      setState(587);
      match(GpuSqlParser::STRING);
      break;
    }

    case 6: {
      setState(588);
      match(GpuSqlParser::DATETIMELIT);
      break;
    }

    case 7: {
      setState(589);
      match(GpuSqlParser::BOOLEANLIT);
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

//----------------- ConstraintContext ------------------------------------------------------------------

GpuSqlParser::ConstraintContext::ConstraintContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::ConstraintContext::UNIQUE() {
  return getToken(GpuSqlParser::UNIQUE, 0);
}

tree::TerminalNode* GpuSqlParser::ConstraintContext::INDEX() {
  return getToken(GpuSqlParser::INDEX, 0);
}

tree::TerminalNode* GpuSqlParser::ConstraintContext::NOT() {
  return getToken(GpuSqlParser::NOT, 0);
}

tree::TerminalNode* GpuSqlParser::ConstraintContext::NULL_T() {
  return getToken(GpuSqlParser::NULL_T, 0);
}


size_t GpuSqlParser::ConstraintContext::getRuleIndex() const {
  return GpuSqlParser::RuleConstraint;
}

void GpuSqlParser::ConstraintContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstraint(this);
}

void GpuSqlParser::ConstraintContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstraint(this);
}

GpuSqlParser::ConstraintContext* GpuSqlParser::constraint() {
  ConstraintContext *_localctx = _tracker.createInstance<ConstraintContext>(_ctx, getState());
  enterRule(_localctx, 136, GpuSqlParser::RuleConstraint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(596);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::UNIQUE: {
        setState(592);
        match(GpuSqlParser::UNIQUE);
        break;
      }

      case GpuSqlParser::INDEX: {
        setState(593);
        match(GpuSqlParser::INDEX);
        break;
      }

      case GpuSqlParser::NOT: {
        setState(594);
        match(GpuSqlParser::NOT);
        setState(595);
        match(GpuSqlParser::NULL_T);
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

//----------------- RetpayloadContext ------------------------------------------------------------------

GpuSqlParser::RetpayloadContext::RetpayloadContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

GpuSqlParser::DatatypeContext* GpuSqlParser::RetpayloadContext::datatype() {
  return getRuleContext<GpuSqlParser::DatatypeContext>(0);
}


size_t GpuSqlParser::RetpayloadContext::getRuleIndex() const {
  return GpuSqlParser::RuleRetpayload;
}

void GpuSqlParser::RetpayloadContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRetpayload(this);
}

void GpuSqlParser::RetpayloadContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRetpayload(this);
}

GpuSqlParser::RetpayloadContext* GpuSqlParser::retpayload() {
  RetpayloadContext *_localctx = _tracker.createInstance<RetpayloadContext>(_ctx, getState());
  enterRule(_localctx, 138, GpuSqlParser::RuleRetpayload);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(598);
    datatype();
   
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

GpuSqlParser::DatatypeContext* GpuSqlParser::CastOperationContext::datatype() {
  return getRuleContext<GpuSqlParser::DatatypeContext>(0);
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

tree::TerminalNode* GpuSqlParser::AggregationContext::LPAREN() {
  return getToken(GpuSqlParser::LPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::RPAREN() {
  return getToken(GpuSqlParser::RPAREN, 0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::MIN_AGG() {
  return getToken(GpuSqlParser::MIN_AGG, 0);
}

GpuSqlParser::ExpressionContext* GpuSqlParser::AggregationContext::expression() {
  return getRuleContext<GpuSqlParser::ExpressionContext>(0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::MAX_AGG() {
  return getToken(GpuSqlParser::MAX_AGG, 0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::SUM_AGG() {
  return getToken(GpuSqlParser::SUM_AGG, 0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::COUNT_AGG() {
  return getToken(GpuSqlParser::COUNT_AGG, 0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::ASTERISK() {
  return getToken(GpuSqlParser::ASTERISK, 0);
}

tree::TerminalNode* GpuSqlParser::AggregationContext::AVG_AGG() {
  return getToken(GpuSqlParser::AVG_AGG, 0);
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

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::ROUND() {
  return getToken(GpuSqlParser::ROUND, 0);
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

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO_LONGITUDE_TO_TILE_X() {
  return getToken(GpuSqlParser::GEO_LONGITUDE_TO_TILE_X, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO_LATITUDE_TO_TILE_Y() {
  return getToken(GpuSqlParser::GEO_LATITUDE_TO_TILE_Y, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO_TILE_X_TO_LONGITUDE() {
  return getToken(GpuSqlParser::GEO_TILE_X_TO_LONGITUDE, 0);
}

tree::TerminalNode* GpuSqlParser::BinaryOperationContext::GEO_TILE_Y_TO_LATITUDE() {
  return getToken(GpuSqlParser::GEO_TILE_Y_TO_LATITUDE, 0);
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

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::LOGICAL_NOT() {
  return getToken(GpuSqlParser::LOGICAL_NOT, 0);
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

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::DATETYPE() {
  return getToken(GpuSqlParser::DATETYPE, 0);
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

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::WEEKDAY() {
  return getToken(GpuSqlParser::WEEKDAY, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::DAYOFWEEK() {
  return getToken(GpuSqlParser::DAYOFWEEK, 0);
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

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::IS() {
  return getToken(GpuSqlParser::IS, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::NULL_T() {
  return getToken(GpuSqlParser::NULL_T, 0);
}

tree::TerminalNode* GpuSqlParser::UnaryOperationContext::NOT() {
  return getToken(GpuSqlParser::NOT, 0);
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
  size_t startState = 140;
  enterRecursionRule(_localctx, 140, GpuSqlParser::RuleExpression, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(926);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 44, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(601);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOGICAL_NOT);
      setState(602);
      expression(82);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(603);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUS);
      setState(604);
      expression(81);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(605);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ABS);
      setState(606);
      match(GpuSqlParser::LPAREN);
      setState(607);
      expression(0);
      setState(608);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(610);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIN);
      setState(611);
      match(GpuSqlParser::LPAREN);
      setState(612);
      expression(0);
      setState(613);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(615);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COS);
      setState(616);
      match(GpuSqlParser::LPAREN);
      setState(617);
      expression(0);
      setState(618);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(620);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::TAN);
      setState(621);
      match(GpuSqlParser::LPAREN);
      setState(622);
      expression(0);
      setState(623);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(625);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::COT);
      setState(626);
      match(GpuSqlParser::LPAREN);
      setState(627);
      expression(0);
      setState(628);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(630);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ASIN);
      setState(631);
      match(GpuSqlParser::LPAREN);
      setState(632);
      expression(0);
      setState(633);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(635);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ACOS);
      setState(636);
      match(GpuSqlParser::LPAREN);
      setState(637);
      expression(0);
      setState(638);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(640);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN);
      setState(641);
      match(GpuSqlParser::LPAREN);
      setState(642);
      expression(0);
      setState(643);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 11: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(645);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG10);
      setState(646);
      match(GpuSqlParser::LPAREN);
      setState(647);
      expression(0);
      setState(648);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 12: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(650);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(651);
      match(GpuSqlParser::LPAREN);
      setState(652);
      expression(0);
      setState(653);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 13: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(655);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::EXP);
      setState(656);
      match(GpuSqlParser::LPAREN);
      setState(657);
      expression(0);
      setState(658);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 14: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(660);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQRT);
      setState(661);
      match(GpuSqlParser::LPAREN);
      setState(662);
      expression(0);
      setState(663);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 15: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(665);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SQUARE);
      setState(666);
      match(GpuSqlParser::LPAREN);
      setState(667);
      expression(0);
      setState(668);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 16: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(670);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SIGN);
      setState(671);
      match(GpuSqlParser::LPAREN);
      setState(672);
      expression(0);
      setState(673);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 17: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(675);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROUND);
      setState(676);
      match(GpuSqlParser::LPAREN);
      setState(677);
      expression(0);
      setState(678);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 18: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(680);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::FLOOR);
      setState(681);
      match(GpuSqlParser::LPAREN);
      setState(682);
      expression(0);
      setState(683);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 19: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(685);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::CEIL);
      setState(686);
      match(GpuSqlParser::LPAREN);
      setState(687);
      expression(0);
      setState(688);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 20: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(690);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DATETYPE);
      setState(691);
      match(GpuSqlParser::LPAREN);
      setState(692);
      expression(0);
      setState(693);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 21: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(695);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::YEAR);
      setState(696);
      match(GpuSqlParser::LPAREN);
      setState(697);
      expression(0);
      setState(698);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 22: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(700);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MONTH);
      setState(701);
      match(GpuSqlParser::LPAREN);
      setState(702);
      expression(0);
      setState(703);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 23: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(705);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DAY);
      setState(706);
      match(GpuSqlParser::LPAREN);
      setState(707);
      expression(0);
      setState(708);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 24: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(710);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::HOUR);
      setState(711);
      match(GpuSqlParser::LPAREN);
      setState(712);
      expression(0);
      setState(713);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 25: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(715);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MINUTE);
      setState(716);
      match(GpuSqlParser::LPAREN);
      setState(717);
      expression(0);
      setState(718);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 26: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(720);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::SECOND);
      setState(721);
      match(GpuSqlParser::LPAREN);
      setState(722);
      expression(0);
      setState(723);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 27: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(725);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::WEEKDAY);
      setState(726);
      match(GpuSqlParser::LPAREN);
      setState(727);
      expression(0);
      setState(728);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 28: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(730);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::DAYOFWEEK);
      setState(731);
      match(GpuSqlParser::LPAREN);
      setState(732);
      expression(0);
      setState(733);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 29: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(735);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LTRIM);
      setState(736);
      match(GpuSqlParser::LPAREN);
      setState(737);
      expression(0);
      setState(738);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 30: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(740);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::RTRIM);
      setState(741);
      match(GpuSqlParser::LPAREN);
      setState(742);
      expression(0);
      setState(743);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 31: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(745);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOWER);
      setState(746);
      match(GpuSqlParser::LPAREN);
      setState(747);
      expression(0);
      setState(748);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 32: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(750);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::UPPER);
      setState(751);
      match(GpuSqlParser::LPAREN);
      setState(752);
      expression(0);
      setState(753);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 33: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(755);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::REVERSE);
      setState(756);
      match(GpuSqlParser::LPAREN);
      setState(757);
      expression(0);
      setState(758);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 34: {
      _localctx = _tracker.createInstance<UnaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(760);
      dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LEN);
      setState(761);
      match(GpuSqlParser::LPAREN);
      setState(762);
      expression(0);
      setState(763);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 35: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(765);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ATAN2);
      setState(766);
      match(GpuSqlParser::LPAREN);
      setState(767);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(768);
      match(GpuSqlParser::COMMA);
      setState(769);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(770);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 36: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(772);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LOG);
      setState(773);
      match(GpuSqlParser::LPAREN);
      setState(774);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(775);
      match(GpuSqlParser::COMMA);
      setState(776);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(777);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 37: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(779);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POW);
      setState(780);
      match(GpuSqlParser::LPAREN);
      setState(781);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(782);
      match(GpuSqlParser::COMMA);
      setState(783);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(784);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 38: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(786);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROOT);
      setState(787);
      match(GpuSqlParser::LPAREN);
      setState(788);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(789);
      match(GpuSqlParser::COMMA);
      setState(790);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(791);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 39: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(793);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::ROUND);
      setState(794);
      match(GpuSqlParser::LPAREN);
      setState(795);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(796);
      match(GpuSqlParser::COMMA);
      setState(797);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(798);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 40: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(800);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::POINT);
      setState(801);
      match(GpuSqlParser::LPAREN);
      setState(802);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(803);
      match(GpuSqlParser::COMMA);
      setState(804);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(805);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 41: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(807);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_CONTAINS);
      setState(808);
      match(GpuSqlParser::LPAREN);
      setState(809);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(810);
      match(GpuSqlParser::COMMA);
      setState(811);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(812);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 42: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(814);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_INTERSECT);
      setState(815);
      match(GpuSqlParser::LPAREN);
      setState(816);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(817);
      match(GpuSqlParser::COMMA);
      setState(818);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(819);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 43: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(821);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_UNION);
      setState(822);
      match(GpuSqlParser::LPAREN);
      setState(823);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(824);
      match(GpuSqlParser::COMMA);
      setState(825);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(826);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 44: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(828);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_LONGITUDE_TO_TILE_X);
      setState(829);
      match(GpuSqlParser::LPAREN);
      setState(830);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(831);
      match(GpuSqlParser::COMMA);
      setState(832);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(833);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 45: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(835);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_LATITUDE_TO_TILE_Y);
      setState(836);
      match(GpuSqlParser::LPAREN);
      setState(837);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(838);
      match(GpuSqlParser::COMMA);
      setState(839);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(840);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 46: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(842);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_TILE_X_TO_LONGITUDE);
      setState(843);
      match(GpuSqlParser::LPAREN);
      setState(844);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(845);
      match(GpuSqlParser::COMMA);
      setState(846);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(847);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 47: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(849);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::GEO_TILE_Y_TO_LATITUDE);
      setState(850);
      match(GpuSqlParser::LPAREN);
      setState(851);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(852);
      match(GpuSqlParser::COMMA);
      setState(853);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(854);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 48: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(856);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::CONCAT);
      setState(857);
      match(GpuSqlParser::LPAREN);
      setState(858);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(859);
      match(GpuSqlParser::COMMA);
      setState(860);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(861);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 49: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(863);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::LEFT);
      setState(864);
      match(GpuSqlParser::LPAREN);
      setState(865);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(866);
      match(GpuSqlParser::COMMA);
      setState(867);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(868);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 50: {
      _localctx = _tracker.createInstance<BinaryOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(870);
      dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::RIGHT);
      setState(871);
      match(GpuSqlParser::LPAREN);
      setState(872);
      dynamic_cast<BinaryOperationContext *>(_localctx)->left = expression(0);
      setState(873);
      match(GpuSqlParser::COMMA);
      setState(874);
      dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(0);
      setState(875);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 51: {
      _localctx = _tracker.createInstance<CastOperationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(877);
      dynamic_cast<CastOperationContext *>(_localctx)->op = match(GpuSqlParser::CAST);
      setState(878);
      match(GpuSqlParser::LPAREN);
      setState(879);
      expression(0);
      setState(880);
      match(GpuSqlParser::AS);
      setState(881);
      datatype();
      setState(882);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 52: {
      _localctx = _tracker.createInstance<ParenExpressionContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(884);
      match(GpuSqlParser::LPAREN);
      setState(885);
      expression(0);
      setState(886);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 53: {
      _localctx = _tracker.createInstance<VarReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(888);
      columnId();
      break;
    }

    case 54: {
      _localctx = _tracker.createInstance<GeoReferenceContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(889);
      geometry();
      break;
    }

    case 55: {
      _localctx = _tracker.createInstance<DateTimeLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(890);
      match(GpuSqlParser::DATETIMELIT);
      break;
    }

    case 56: {
      _localctx = _tracker.createInstance<DecimalLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(891);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 57: {
      _localctx = _tracker.createInstance<PiLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(892);
      match(GpuSqlParser::PI);
      break;
    }

    case 58: {
      _localctx = _tracker.createInstance<NowLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(893);
      match(GpuSqlParser::NOW);
      break;
    }

    case 59: {
      _localctx = _tracker.createInstance<IntLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(894);
      match(GpuSqlParser::INTLIT);
      break;
    }

    case 60: {
      _localctx = _tracker.createInstance<StringLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(895);
      match(GpuSqlParser::STRING);
      break;
    }

    case 61: {
      _localctx = _tracker.createInstance<BooleanLiteralContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(896);
      match(GpuSqlParser::BOOLEANLIT);
      break;
    }

    case 62: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(897);
      dynamic_cast<AggregationContext *>(_localctx)->op = match(GpuSqlParser::MIN_AGG);
      setState(898);
      match(GpuSqlParser::LPAREN);

      setState(899);
      expression(0);
      setState(900);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 63: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(902);
      dynamic_cast<AggregationContext *>(_localctx)->op = match(GpuSqlParser::MAX_AGG);
      setState(903);
      match(GpuSqlParser::LPAREN);

      setState(904);
      expression(0);
      setState(905);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 64: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(907);
      dynamic_cast<AggregationContext *>(_localctx)->op = match(GpuSqlParser::SUM_AGG);
      setState(908);
      match(GpuSqlParser::LPAREN);

      setState(909);
      expression(0);
      setState(910);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 65: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(912);
      dynamic_cast<AggregationContext *>(_localctx)->op = match(GpuSqlParser::COUNT_AGG);
      setState(913);
      match(GpuSqlParser::LPAREN);

      setState(914);
      expression(0);
      setState(915);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 66: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(917);
      dynamic_cast<AggregationContext *>(_localctx)->op = match(GpuSqlParser::COUNT_AGG);
      setState(918);
      match(GpuSqlParser::LPAREN);
      setState(919);
      match(GpuSqlParser::ASTERISK);
      setState(920);
      match(GpuSqlParser::RPAREN);
      break;
    }

    case 67: {
      _localctx = _tracker.createInstance<AggregationContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(921);
      dynamic_cast<AggregationContext *>(_localctx)->op = match(GpuSqlParser::AVG_AGG);
      setState(922);
      match(GpuSqlParser::LPAREN);

      setState(923);
      expression(0);
      setState(924);
      match(GpuSqlParser::RPAREN);
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(979);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 46, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(977);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 45, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(928);

          if (!(precpred(_ctx, 46))) throw FailedPredicateException(this, "precpred(_ctx, 46)");
          setState(929);
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
          setState(930);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(47);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(931);

          if (!(precpred(_ctx, 45))) throw FailedPredicateException(this, "precpred(_ctx, 45)");
          setState(932);
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
          setState(933);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(46);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(934);

          if (!(precpred(_ctx, 44))) throw FailedPredicateException(this, "precpred(_ctx, 44)");
          setState(935);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::MODULO);
          setState(936);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(45);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(937);

          if (!(precpred(_ctx, 38))) throw FailedPredicateException(this, "precpred(_ctx, 38)");
          setState(938);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::XOR);
          setState(939);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(39);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(940);

          if (!(precpred(_ctx, 37))) throw FailedPredicateException(this, "precpred(_ctx, 37)");
          setState(941);
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
          setState(942);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(38);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(943);

          if (!(precpred(_ctx, 36))) throw FailedPredicateException(this, "precpred(_ctx, 36)");
          setState(944);
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
          setState(945);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(37);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(946);

          if (!(precpred(_ctx, 35))) throw FailedPredicateException(this, "precpred(_ctx, 35)");
          setState(947);
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
          setState(948);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(36);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(949);

          if (!(precpred(_ctx, 34))) throw FailedPredicateException(this, "precpred(_ctx, 34)");
          setState(950);
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
          setState(951);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(35);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(952);

          if (!(precpred(_ctx, 33))) throw FailedPredicateException(this, "precpred(_ctx, 33)");
          setState(953);
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
          setState(954);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(34);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(955);

          if (!(precpred(_ctx, 32))) throw FailedPredicateException(this, "precpred(_ctx, 32)");
          setState(956);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOTEQUALS_GT_LT);
          setState(957);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(33);
          break;
        }

        case 11: {
          auto newContext = _tracker.createInstance<TernaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(958);

          if (!(precpred(_ctx, 20))) throw FailedPredicateException(this, "precpred(_ctx, 20)");
          setState(959);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op = match(GpuSqlParser::BETWEEN);
          setState(960);
          expression(0);
          setState(961);
          dynamic_cast<TernaryOperationContext *>(_localctx)->op2 = match(GpuSqlParser::AND);
          setState(962);
          expression(21);
          break;
        }

        case 12: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(964);

          if (!(precpred(_ctx, 19))) throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(965);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::AND);
          setState(966);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(20);
          break;
        }

        case 13: {
          auto newContext = _tracker.createInstance<BinaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          newContext->left = previousContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(967);

          if (!(precpred(_ctx, 18))) throw FailedPredicateException(this, "precpred(_ctx, 18)");
          setState(968);
          dynamic_cast<BinaryOperationContext *>(_localctx)->op = match(GpuSqlParser::OR);
          setState(969);
          dynamic_cast<BinaryOperationContext *>(_localctx)->right = expression(19);
          break;
        }

        case 14: {
          auto newContext = _tracker.createInstance<UnaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(970);

          if (!(precpred(_ctx, 80))) throw FailedPredicateException(this, "precpred(_ctx, 80)");
          setState(971);
          match(GpuSqlParser::IS);
          setState(972);
          dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NULL_T);
          break;
        }

        case 15: {
          auto newContext = _tracker.createInstance<UnaryOperationContext>(_tracker.createInstance<ExpressionContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpression);
          setState(973);

          if (!(precpred(_ctx, 79))) throw FailedPredicateException(this, "precpred(_ctx, 79)");
          setState(974);
          match(GpuSqlParser::IS);
          setState(975);
          dynamic_cast<UnaryOperationContext *>(_localctx)->op = match(GpuSqlParser::NOT);
          setState(976);
          match(GpuSqlParser::NULL_T);
          break;
        }

        } 
      }
      setState(981);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 46, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- DatatypeContext ------------------------------------------------------------------

GpuSqlParser::DatatypeContext::DatatypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::INTTYPE() {
  return getToken(GpuSqlParser::INTTYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::LONGTYPE() {
  return getToken(GpuSqlParser::LONGTYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::DATETYPE() {
  return getToken(GpuSqlParser::DATETYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::DETETIMETYPE() {
  return getToken(GpuSqlParser::DETETIMETYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::FLOATTYPE() {
  return getToken(GpuSqlParser::FLOATTYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::DOUBLETYPE() {
  return getToken(GpuSqlParser::DOUBLETYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::STRINGTYPE() {
  return getToken(GpuSqlParser::STRINGTYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::BOOLEANTYPE() {
  return getToken(GpuSqlParser::BOOLEANTYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::POINTTYPE() {
  return getToken(GpuSqlParser::POINTTYPE, 0);
}

tree::TerminalNode* GpuSqlParser::DatatypeContext::POLYTYPE() {
  return getToken(GpuSqlParser::POLYTYPE, 0);
}


size_t GpuSqlParser::DatatypeContext::getRuleIndex() const {
  return GpuSqlParser::RuleDatatype;
}

void GpuSqlParser::DatatypeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDatatype(this);
}

void GpuSqlParser::DatatypeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<GpuSqlParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDatatype(this);
}

GpuSqlParser::DatatypeContext* GpuSqlParser::datatype() {
  DatatypeContext *_localctx = _tracker.createInstance<DatatypeContext>(_ctx, getState());
  enterRule(_localctx, 142, GpuSqlParser::RuleDatatype);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(982);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << GpuSqlParser::INTTYPE)
      | (1ULL << GpuSqlParser::LONGTYPE)
      | (1ULL << GpuSqlParser::DATETYPE)
      | (1ULL << GpuSqlParser::DETETIMETYPE)
      | (1ULL << GpuSqlParser::FLOATTYPE)
      | (1ULL << GpuSqlParser::DOUBLETYPE)
      | (1ULL << GpuSqlParser::STRINGTYPE)
      | (1ULL << GpuSqlParser::BOOLEANTYPE)
      | (1ULL << GpuSqlParser::POINTTYPE)
      | (1ULL << GpuSqlParser::POLYTYPE))) != 0))) {
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
  enterRule(_localctx, 144, GpuSqlParser::RuleGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(990);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::POINT: {
        setState(984);
        pointGeometry();
        break;
      }

      case GpuSqlParser::POLYGON: {
        setState(985);
        polygonGeometry();
        break;
      }

      case GpuSqlParser::LINESTRING: {
        setState(986);
        lineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOINT: {
        setState(987);
        multiPointGeometry();
        break;
      }

      case GpuSqlParser::MULTILINESTRING: {
        setState(988);
        multiLineStringGeometry();
        break;
      }

      case GpuSqlParser::MULTIPOLYGON: {
        setState(989);
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
  enterRule(_localctx, 146, GpuSqlParser::RulePointGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(992);
    match(GpuSqlParser::POINT);
    setState(993);
    match(GpuSqlParser::LPAREN);
    setState(994);
    point();
    setState(995);
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
  enterRule(_localctx, 148, GpuSqlParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(997);
    match(GpuSqlParser::LINESTRING);
    setState(998);
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
  enterRule(_localctx, 150, GpuSqlParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1000);
    match(GpuSqlParser::POLYGON);
    setState(1001);
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
  enterRule(_localctx, 152, GpuSqlParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1003);
    match(GpuSqlParser::MULTIPOINT);
    setState(1004);
    match(GpuSqlParser::LPAREN);
    setState(1005);
    pointOrClosedPoint();
    setState(1010);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(1006);
      match(GpuSqlParser::COMMA);
      setState(1007);
      pointOrClosedPoint();
      setState(1012);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1013);
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
  enterRule(_localctx, 154, GpuSqlParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1015);
    match(GpuSqlParser::MULTILINESTRING);
    setState(1016);
    match(GpuSqlParser::LPAREN);
    setState(1017);
    lineString();
    setState(1022);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(1018);
      match(GpuSqlParser::COMMA);
      setState(1019);
      lineString();
      setState(1024);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1025);
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
  enterRule(_localctx, 156, GpuSqlParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1027);
    match(GpuSqlParser::MULTIPOLYGON);
    setState(1028);
    match(GpuSqlParser::LPAREN);
    setState(1029);
    polygon();
    setState(1034);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(1030);
      match(GpuSqlParser::COMMA);
      setState(1031);
      polygon();
      setState(1036);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1037);
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
  enterRule(_localctx, 158, GpuSqlParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(1044);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case GpuSqlParser::MINUS:
      case GpuSqlParser::FLOATLIT:
      case GpuSqlParser::INTLIT: {
        enterOuterAlt(_localctx, 1);
        setState(1039);
        point();
        break;
      }

      case GpuSqlParser::LPAREN: {
        enterOuterAlt(_localctx, 2);
        setState(1040);
        match(GpuSqlParser::LPAREN);
        setState(1041);
        point();
        setState(1042);
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
  enterRule(_localctx, 160, GpuSqlParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1046);
    match(GpuSqlParser::LPAREN);
    setState(1047);
    lineString();
    setState(1052);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(1048);
      match(GpuSqlParser::COMMA);
      setState(1049);
      lineString();
      setState(1054);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1055);
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
  enterRule(_localctx, 162, GpuSqlParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1057);
    match(GpuSqlParser::LPAREN);
    setState(1058);
    point();
    setState(1063);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == GpuSqlParser::COMMA) {
      setState(1059);
      match(GpuSqlParser::COMMA);
      setState(1060);
      point();
      setState(1065);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(1066);
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

std::vector<tree::TerminalNode *> GpuSqlParser::PointContext::MINUS() {
  return getTokens(GpuSqlParser::MINUS);
}

tree::TerminalNode* GpuSqlParser::PointContext::MINUS(size_t i) {
  return getToken(GpuSqlParser::MINUS, i);
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
  enterRule(_localctx, 164, GpuSqlParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(1076);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 56, _ctx)) {
    case 1: {
      setState(1069);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == GpuSqlParser::MINUS) {
        setState(1068);
        match(GpuSqlParser::MINUS);
      }
      setState(1071);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 2: {
      setState(1073);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == GpuSqlParser::MINUS) {
        setState(1072);
        match(GpuSqlParser::MINUS);
      }
      setState(1075);
      match(GpuSqlParser::INTLIT);
      break;
    }

    }
    setState(1086);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 59, _ctx)) {
    case 1: {
      setState(1079);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == GpuSqlParser::MINUS) {
        setState(1078);
        match(GpuSqlParser::MINUS);
      }
      setState(1081);
      match(GpuSqlParser::FLOATLIT);
      break;
    }

    case 2: {
      setState(1083);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == GpuSqlParser::MINUS) {
        setState(1082);
        match(GpuSqlParser::MINUS);
      }
      setState(1085);
      match(GpuSqlParser::INTLIT);
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

bool GpuSqlParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 70: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool GpuSqlParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 46);
    case 1: return precpred(_ctx, 45);
    case 2: return precpred(_ctx, 44);
    case 3: return precpred(_ctx, 38);
    case 4: return precpred(_ctx, 37);
    case 5: return precpred(_ctx, 36);
    case 6: return precpred(_ctx, 35);
    case 7: return precpred(_ctx, 34);
    case 8: return precpred(_ctx, 33);
    case 9: return precpred(_ctx, 32);
    case 10: return precpred(_ctx, 20);
    case 11: return precpred(_ctx, 19);
    case 12: return precpred(_ctx, 18);
    case 13: return precpred(_ctx, 80);
    case 14: return precpred(_ctx, 79);

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
  "showColumns", "showConstraints", "showQueryTypes", "sqlSelect", "sqlCreateDb", 
  "sqlDropDb", "sqlCreateTable", "sqlDropTable", "sqlAlterTable", "sqlAlterDatabase", 
  "sqlCreateIndex", "sqlInsertInto", "newTableEntries", "newTableEntry", 
  "alterDatabaseEntries", "alterDatabaseEntry", "renameDatabase", "alterTableEntries", 
  "alterTableEntry", "addColumn", "dropColumn", "alterColumn", "renameColumn", 
  "renameTable", "addConstraint", "dropConstraint", "alterBlockSize", "renameColumnFrom", 
  "renameColumnTo", "newTableColumn", "newTableConstraint", "selectColumns", 
  "selectColumn", "selectAllColumns", "whereClause", "orderByColumns", "orderByColumn", 
  "insertIntoValues", "insertIntoColumns", "indexColumns", "constraintColumns", 
  "groupByColumns", "groupByColumn", "fromTables", "joinClauses", "joinClause", 
  "joinTable", "joinColumnLeft", "joinColumnRight", "joinOperator", "joinType", 
  "fromTable", "columnId", "table", "column", "database", "alias", "indexName", 
  "constraintName", "limit", "offset", "blockSize", "columnValue", "constraint", 
  "retpayload", "expression", "datatype", "geometry", "pointGeometry", "lineStringGeometry", 
  "polygonGeometry", "multiPointGeometry", "multiLineStringGeometry", "multiPolygonGeometry", 
  "pointOrClosedPoint", "polygon", "lineString", "point"
};

std::vector<std::string> GpuSqlParser::_literalNames = {
  "", "", "'\n'", "'\r'", "'\r\n'", "", "';'", "'''", "'\"'", "'_'", "':'", 
  "','", "'.'", "'['", "']'", "", "", "", "", "'POINT'", "'MULTIPOINT'", 
  "'LINESTRING'", "'MULTILINESTRING'", "'POLYGON'", "'MULTIPOLYGON'", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "'+'", "'-'", "'*'", "'/'", "'%'", "'^'", "'='", "'!='", "'<>'", "'('", 
  "')'", "'>'", "'<'", "'>='", "'<='", "'!'", "", "", "'|'", "'&'", "'<<'", 
  "'>>'"
};

std::vector<std::string> GpuSqlParser::_symbolicNames = {
  "", "DATETIMELIT", "LF", "CR", "CRLF", "WS", "SEMICOL", "SQOUTE", "DQOUTE", 
  "UNDERSCORE", "COLON", "COMMA", "DOT", "LSQR_BRC", "RSQR_BRC", "STRING", 
  "DELIMID", "DATELIT", "TIMELIT", "POINT", "MULTIPOINT", "LINESTRING", 
  "MULTILINESTRING", "POLYGON", "MULTIPOLYGON", "INTTYPE", "LONGTYPE", "DATETYPE", 
  "DETETIMETYPE", "FLOATTYPE", "DOUBLETYPE", "STRINGTYPE", "BOOLEANTYPE", 
  "POINTTYPE", "POLYTYPE", "TYPES", "INDEX", "UNIQUE", "PRIMARY", "KEY", 
  "CREATE", "ADD", "DROP", "ALTER", "RENAME", "SET", "INSERT", "INTO", "SHOW", 
  "SIZE", "FULL", "OUTER", "INNER", "DATABASE", "DATABASES", "TABLE", "TABLES", 
  "COLUMN", "COLUMNS", "BLOCK", "CONSTRAINTS", "VALUES", "SELECT", "FROM", 
  "JOIN", "WHERE", "GROUP", "AS", "IN", "TO", "IS", "NOT", "NULL_T", "BY", 
  "BETWEEN", "ON", "ORDER", "DIR", "LIMIT", "OFFSET", "QUERY", "AVG_AGG", 
  "SUM_AGG", "MIN_AGG", "MAX_AGG", "COUNT_AGG", "YEAR", "MONTH", "DAY", 
  "HOUR", "MINUTE", "SECOND", "WEEKDAY", "DAYOFWEEK", "NOW", "PI", "ABS", 
  "SIN", "COS", "TAN", "COT", "ASIN", "ACOS", "ATAN", "ATAN2", "LOG10", 
  "LOG", "EXP", "POW", "SQRT", "SQUARE", "SIGN", "ROOT", "ROUND", "CEIL", 
  "FLOOR", "LTRIM", "RTRIM", "LOWER", "UPPER", "REVERSE", "LEN", "LEFT", 
  "RIGHT", "CONCAT", "CAST", "RETPAYLOAD", "GEO_CONTAINS", "GEO_INTERSECT", 
  "GEO_UNION", "GEO_LONGITUDE_TO_TILE_X", "GEO_LATITUDE_TO_TILE_Y", "GEO_TILE_X_TO_LONGITUDE", 
  "GEO_TILE_Y_TO_LATITUDE", "PLUS", "MINUS", "ASTERISK", "DIVISION", "MODULO", 
  "XOR", "EQUALS", "NOTEQUALS", "NOTEQUALS_GT_LT", "LPAREN", "RPAREN", "GREATER", 
  "LESS", "GREATEREQ", "LESSEQ", "LOGICAL_NOT", "OR", "AND", "BIT_OR", "BIT_AND", 
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
    0x3, 0xa3, 0x443, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x4, 0x43, 0x9, 0x43, 0x4, 0x44, 0x9, 0x44, 0x4, 0x45, 0x9, 0x45, 0x4, 
    0x46, 0x9, 0x46, 0x4, 0x47, 0x9, 0x47, 0x4, 0x48, 0x9, 0x48, 0x4, 0x49, 
    0x9, 0x49, 0x4, 0x4a, 0x9, 0x4a, 0x4, 0x4b, 0x9, 0x4b, 0x4, 0x4c, 0x9, 
    0x4c, 0x4, 0x4d, 0x9, 0x4d, 0x4, 0x4e, 0x9, 0x4e, 0x4, 0x4f, 0x9, 0x4f, 
    0x4, 0x50, 0x9, 0x50, 0x4, 0x51, 0x9, 0x51, 0x4, 0x52, 0x9, 0x52, 0x4, 
    0x53, 0x9, 0x53, 0x4, 0x54, 0x9, 0x54, 0x3, 0x2, 0x7, 0x2, 0xaa, 0xa, 
    0x2, 0xc, 0x2, 0xe, 0x2, 0xad, 0xb, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0xbc, 0xa, 0x3, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 0xc2, 0xa, 0x4, 0x3, 0x5, 0x3, 0x5, 
    0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 
    0xcc, 0xa, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0xd6, 0xa, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 
    0xe0, 0xa, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x5, 0xa, 0xef, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xf3, 0xa, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xf8, 0xa, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xfd, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 
    0xa, 0x101, 0xa, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0x105, 0xa, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 
    0x10d, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x11a, 
    0xa, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 
    0x13, 0x3, 0x13, 0x7, 0x13, 0x14b, 0xa, 0x13, 0xc, 0x13, 0xe, 0x13, 
    0x14e, 0xb, 0x13, 0x3, 0x14, 0x3, 0x14, 0x5, 0x14, 0x152, 0xa, 0x14, 
    0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x7, 0x15, 0x157, 0xa, 0x15, 0xc, 0x15, 
    0xe, 0x15, 0x15a, 0xb, 0x15, 0x3, 0x16, 0x3, 0x16, 0x5, 0x16, 0x15e, 
    0xa, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 
    0x18, 0x3, 0x18, 0x7, 0x18, 0x167, 0xa, 0x18, 0xc, 0x18, 0xe, 0x18, 
    0x16a, 0xb, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 
    0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x5, 0x19, 0x174, 0xa, 0x19, 0x3, 0x1a, 
    0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 
    0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1d, 
    0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1e, 0x3, 
    0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 
    0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 
    0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x5, 
    0x24, 0x1a4, 0xa, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 
    0x3, 0x25, 0x3, 0x25, 0x3, 0x26, 0x3, 0x26, 0x5, 0x26, 0x1ae, 0xa, 0x26, 
    0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x5, 0x26, 0x1b3, 0xa, 0x26, 0x7, 0x26, 
    0x1b5, 0xa, 0x26, 0xc, 0x26, 0xe, 0x26, 0x1b8, 0xb, 0x26, 0x3, 0x27, 
    0x3, 0x27, 0x3, 0x27, 0x5, 0x27, 0x1bd, 0xa, 0x27, 0x3, 0x27, 0x3, 0x27, 
    0x5, 0x27, 0x1c1, 0xa, 0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x29, 0x3, 0x29, 
    0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x7, 0x2a, 0x1ca, 0xa, 0x2a, 0xc, 0x2a, 
    0xe, 0x2a, 0x1cd, 0xb, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 0x1d1, 
    0xa, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x7, 0x2c, 0x1d6, 0xa, 0x2c, 
    0xc, 0x2c, 0xe, 0x2c, 0x1d9, 0xb, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 
    0x7, 0x2d, 0x1de, 0xa, 0x2d, 0xc, 0x2d, 0xe, 0x2d, 0x1e1, 0xb, 0x2d, 
    0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x7, 0x2e, 0x1e6, 0xa, 0x2e, 0xc, 0x2e, 
    0xe, 0x2e, 0x1e9, 0xb, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x7, 0x2f, 
    0x1ee, 0xa, 0x2f, 0xc, 0x2f, 0xe, 0x2f, 0x1f1, 0xb, 0x2f, 0x3, 0x30, 
    0x3, 0x30, 0x3, 0x30, 0x7, 0x30, 0x1f6, 0xa, 0x30, 0xc, 0x30, 0xe, 0x30, 
    0x1f9, 0xb, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x7, 0x32, 0x200, 0xa, 0x32, 0xc, 0x32, 0xe, 0x32, 0x203, 0xb, 0x32, 
    0x3, 0x33, 0x6, 0x33, 0x206, 0xa, 0x33, 0xd, 0x33, 0xe, 0x33, 0x207, 
    0x3, 0x34, 0x5, 0x34, 0x20b, 0xa, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 
    0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x35, 0x3, 0x35, 0x3, 
    0x35, 0x5, 0x35, 0x217, 0xa, 0x35, 0x3, 0x36, 0x3, 0x36, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x38, 0x3, 0x38, 0x3, 0x39, 0x3, 0x39, 0x3, 0x39, 0x3, 
    0x39, 0x3, 0x39, 0x5, 0x39, 0x224, 0xa, 0x39, 0x3, 0x3a, 0x3, 0x3a, 
    0x3, 0x3a, 0x5, 0x3a, 0x229, 0xa, 0x3a, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 
    0x3, 0x3b, 0x3, 0x3b, 0x5, 0x3b, 0x230, 0xa, 0x3b, 0x3, 0x3c, 0x3, 0x3c, 
    0x3, 0x3d, 0x3, 0x3d, 0x3, 0x3e, 0x3, 0x3e, 0x3, 0x3f, 0x3, 0x3f, 0x3, 
    0x40, 0x3, 0x40, 0x3, 0x41, 0x3, 0x41, 0x3, 0x42, 0x3, 0x42, 0x3, 0x43, 
    0x3, 0x43, 0x3, 0x44, 0x3, 0x44, 0x3, 0x45, 0x5, 0x45, 0x245, 0xa, 0x45, 
    0x3, 0x45, 0x3, 0x45, 0x5, 0x45, 0x249, 0xa, 0x45, 0x3, 0x45, 0x3, 0x45, 
    0x3, 0x45, 0x3, 0x45, 0x3, 0x45, 0x3, 0x45, 0x5, 0x45, 0x251, 0xa, 0x45, 
    0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x3, 0x46, 0x5, 0x46, 0x257, 0xa, 0x46, 
    0x3, 0x47, 0x3, 0x47, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x5, 0x48, 0x3a1, 0xa, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 
    0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 0x48, 0x3, 
    0x48, 0x3, 0x48, 0x3, 0x48, 0x7, 0x48, 0x3d4, 0xa, 0x48, 0xc, 0x48, 
    0xe, 0x48, 0x3d7, 0xb, 0x48, 0x3, 0x49, 0x3, 0x49, 0x3, 0x4a, 0x3, 0x4a, 
    0x3, 0x4a, 0x3, 0x4a, 0x3, 0x4a, 0x3, 0x4a, 0x5, 0x4a, 0x3e1, 0xa, 0x4a, 
    0x3, 0x4b, 0x3, 0x4b, 0x3, 0x4b, 0x3, 0x4b, 0x3, 0x4b, 0x3, 0x4c, 0x3, 
    0x4c, 0x3, 0x4c, 0x3, 0x4d, 0x3, 0x4d, 0x3, 0x4d, 0x3, 0x4e, 0x3, 0x4e, 
    0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x7, 0x4e, 0x3f3, 0xa, 0x4e, 0xc, 0x4e, 
    0xe, 0x4e, 0x3f6, 0xb, 0x4e, 0x3, 0x4e, 0x3, 0x4e, 0x3, 0x4f, 0x3, 0x4f, 
    0x3, 0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x7, 0x4f, 0x3ff, 0xa, 0x4f, 0xc, 0x4f, 
    0xe, 0x4f, 0x402, 0xb, 0x4f, 0x3, 0x4f, 0x3, 0x4f, 0x3, 0x50, 0x3, 0x50, 
    0x3, 0x50, 0x3, 0x50, 0x3, 0x50, 0x7, 0x50, 0x40b, 0xa, 0x50, 0xc, 0x50, 
    0xe, 0x50, 0x40e, 0xb, 0x50, 0x3, 0x50, 0x3, 0x50, 0x3, 0x51, 0x3, 0x51, 
    0x3, 0x51, 0x3, 0x51, 0x3, 0x51, 0x5, 0x51, 0x417, 0xa, 0x51, 0x3, 0x52, 
    0x3, 0x52, 0x3, 0x52, 0x3, 0x52, 0x7, 0x52, 0x41d, 0xa, 0x52, 0xc, 0x52, 
    0xe, 0x52, 0x420, 0xb, 0x52, 0x3, 0x52, 0x3, 0x52, 0x3, 0x53, 0x3, 0x53, 
    0x3, 0x53, 0x3, 0x53, 0x7, 0x53, 0x428, 0xa, 0x53, 0xc, 0x53, 0xe, 0x53, 
    0x42b, 0xb, 0x53, 0x3, 0x53, 0x3, 0x53, 0x3, 0x54, 0x5, 0x54, 0x430, 
    0xa, 0x54, 0x3, 0x54, 0x3, 0x54, 0x5, 0x54, 0x434, 0xa, 0x54, 0x3, 0x54, 
    0x5, 0x54, 0x437, 0xa, 0x54, 0x3, 0x54, 0x5, 0x54, 0x43a, 0xa, 0x54, 
    0x3, 0x54, 0x3, 0x54, 0x5, 0x54, 0x43e, 0xa, 0x54, 0x3, 0x54, 0x5, 0x54, 
    0x441, 0xa, 0x54, 0x3, 0x54, 0x2, 0x3, 0x8e, 0x55, 0x2, 0x4, 0x6, 0x8, 
    0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 
    0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 
    0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 
    0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 
    0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e, 0x80, 
    0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 
    0x9a, 0x9c, 0x9e, 0xa0, 0xa2, 0xa4, 0xa6, 0x2, 0xd, 0x4, 0x2, 0x41, 
    0x41, 0x46, 0x46, 0x4, 0x2, 0x8e, 0x90, 0x93, 0x96, 0x4, 0x2, 0x12, 
    0x12, 0xa3, 0xa3, 0x3, 0x2, 0x8a, 0x8b, 0x3, 0x2, 0x88, 0x89, 0x3, 0x2, 
    0x9a, 0x9b, 0x3, 0x2, 0x9c, 0x9d, 0x3, 0x2, 0x93, 0x94, 0x3, 0x2, 0x95, 
    0x96, 0x3, 0x2, 0x8e, 0x8f, 0x3, 0x2, 0x1b, 0x24, 0x2, 0x496, 0x2, 0xab, 
    0x3, 0x2, 0x2, 0x2, 0x4, 0xbb, 0x3, 0x2, 0x2, 0x2, 0x6, 0xc1, 0x3, 0x2, 
    0x2, 0x2, 0x8, 0xc3, 0x3, 0x2, 0x2, 0x2, 0xa, 0xc7, 0x3, 0x2, 0x2, 0x2, 
    0xc, 0xcf, 0x3, 0x2, 0x2, 0x2, 0xe, 0xd9, 0x3, 0x2, 0x2, 0x2, 0x10, 
    0xe3, 0x3, 0x2, 0x2, 0x2, 0x12, 0xe9, 0x3, 0x2, 0x2, 0x2, 0x14, 0x108, 
    0x3, 0x2, 0x2, 0x2, 0x16, 0x110, 0x3, 0x2, 0x2, 0x2, 0x18, 0x115, 0x3, 
    0x2, 0x2, 0x2, 0x1a, 0x120, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x125, 0x3, 0x2, 
    0x2, 0x2, 0x1e, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x20, 0x131, 0x3, 0x2, 0x2, 
    0x2, 0x22, 0x13b, 0x3, 0x2, 0x2, 0x2, 0x24, 0x147, 0x3, 0x2, 0x2, 0x2, 
    0x26, 0x151, 0x3, 0x2, 0x2, 0x2, 0x28, 0x153, 0x3, 0x2, 0x2, 0x2, 0x2a, 
    0x15d, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x15f, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x163, 
    0x3, 0x2, 0x2, 0x2, 0x30, 0x173, 0x3, 0x2, 0x2, 0x2, 0x32, 0x175, 0x3, 
    0x2, 0x2, 0x2, 0x34, 0x179, 0x3, 0x2, 0x2, 0x2, 0x36, 0x17d, 0x3, 0x2, 
    0x2, 0x2, 0x38, 0x182, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x188, 0x3, 0x2, 0x2, 
    0x2, 0x3c, 0x18c, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x193, 0x3, 0x2, 0x2, 0x2, 
    0x40, 0x197, 0x3, 0x2, 0x2, 0x2, 0x42, 0x19c, 0x3, 0x2, 0x2, 0x2, 0x44, 
    0x19e, 0x3, 0x2, 0x2, 0x2, 0x46, 0x1a0, 0x3, 0x2, 0x2, 0x2, 0x48, 0x1a5, 
    0x3, 0x2, 0x2, 0x2, 0x4a, 0x1ad, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x1b9, 0x3, 
    0x2, 0x2, 0x2, 0x4e, 0x1c2, 0x3, 0x2, 0x2, 0x2, 0x50, 0x1c4, 0x3, 0x2, 
    0x2, 0x2, 0x52, 0x1c6, 0x3, 0x2, 0x2, 0x2, 0x54, 0x1ce, 0x3, 0x2, 0x2, 
    0x2, 0x56, 0x1d2, 0x3, 0x2, 0x2, 0x2, 0x58, 0x1da, 0x3, 0x2, 0x2, 0x2, 
    0x5a, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x1ea, 0x3, 0x2, 0x2, 0x2, 0x5e, 
    0x1f2, 0x3, 0x2, 0x2, 0x2, 0x60, 0x1fa, 0x3, 0x2, 0x2, 0x2, 0x62, 0x1fc, 
    0x3, 0x2, 0x2, 0x2, 0x64, 0x205, 0x3, 0x2, 0x2, 0x2, 0x66, 0x20a, 0x3, 
    0x2, 0x2, 0x2, 0x68, 0x213, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x218, 0x3, 0x2, 
    0x2, 0x2, 0x6c, 0x21a, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x21c, 0x3, 0x2, 0x2, 
    0x2, 0x70, 0x223, 0x3, 0x2, 0x2, 0x2, 0x72, 0x225, 0x3, 0x2, 0x2, 0x2, 
    0x74, 0x22f, 0x3, 0x2, 0x2, 0x2, 0x76, 0x231, 0x3, 0x2, 0x2, 0x2, 0x78, 
    0x233, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x235, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x237, 
    0x3, 0x2, 0x2, 0x2, 0x7e, 0x239, 0x3, 0x2, 0x2, 0x2, 0x80, 0x23b, 0x3, 
    0x2, 0x2, 0x2, 0x82, 0x23d, 0x3, 0x2, 0x2, 0x2, 0x84, 0x23f, 0x3, 0x2, 
    0x2, 0x2, 0x86, 0x241, 0x3, 0x2, 0x2, 0x2, 0x88, 0x250, 0x3, 0x2, 0x2, 
    0x2, 0x8a, 0x256, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x258, 0x3, 0x2, 0x2, 0x2, 
    0x8e, 0x3a0, 0x3, 0x2, 0x2, 0x2, 0x90, 0x3d8, 0x3, 0x2, 0x2, 0x2, 0x92, 
    0x3e0, 0x3, 0x2, 0x2, 0x2, 0x94, 0x3e2, 0x3, 0x2, 0x2, 0x2, 0x96, 0x3e7, 
    0x3, 0x2, 0x2, 0x2, 0x98, 0x3ea, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x3ed, 0x3, 
    0x2, 0x2, 0x2, 0x9c, 0x3f9, 0x3, 0x2, 0x2, 0x2, 0x9e, 0x405, 0x3, 0x2, 
    0x2, 0x2, 0xa0, 0x416, 0x3, 0x2, 0x2, 0x2, 0xa2, 0x418, 0x3, 0x2, 0x2, 
    0x2, 0xa4, 0x423, 0x3, 0x2, 0x2, 0x2, 0xa6, 0x436, 0x3, 0x2, 0x2, 0x2, 
    0xa8, 0xaa, 0x5, 0x4, 0x3, 0x2, 0xa9, 0xa8, 0x3, 0x2, 0x2, 0x2, 0xaa, 
    0xad, 0x3, 0x2, 0x2, 0x2, 0xab, 0xa9, 0x3, 0x2, 0x2, 0x2, 0xab, 0xac, 
    0x3, 0x2, 0x2, 0x2, 0xac, 0xae, 0x3, 0x2, 0x2, 0x2, 0xad, 0xab, 0x3, 
    0x2, 0x2, 0x2, 0xae, 0xaf, 0x7, 0x2, 0x2, 0x3, 0xaf, 0x3, 0x3, 0x2, 
    0x2, 0x2, 0xb0, 0xbc, 0x5, 0x12, 0xa, 0x2, 0xb1, 0xbc, 0x5, 0x14, 0xb, 
    0x2, 0xb2, 0xbc, 0x5, 0x16, 0xc, 0x2, 0xb3, 0xbc, 0x5, 0x18, 0xd, 0x2, 
    0xb4, 0xbc, 0x5, 0x1a, 0xe, 0x2, 0xb5, 0xbc, 0x5, 0x1c, 0xf, 0x2, 0xb6, 
    0xbc, 0x5, 0x1e, 0x10, 0x2, 0xb7, 0xbc, 0x5, 0x20, 0x11, 0x2, 0xb8, 
    0xbc, 0x5, 0x22, 0x12, 0x2, 0xb9, 0xbc, 0x5, 0x6, 0x4, 0x2, 0xba, 0xbc, 
    0x5, 0x10, 0x9, 0x2, 0xbb, 0xb0, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xb1, 0x3, 
    0x2, 0x2, 0x2, 0xbb, 0xb2, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xb3, 0x3, 0x2, 
    0x2, 0x2, 0xbb, 0xb4, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xb5, 0x3, 0x2, 0x2, 
    0x2, 0xbb, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xb7, 0x3, 0x2, 0x2, 0x2, 
    0xbb, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xb9, 0x3, 0x2, 0x2, 0x2, 0xbb, 
    0xba, 0x3, 0x2, 0x2, 0x2, 0xbc, 0x5, 0x3, 0x2, 0x2, 0x2, 0xbd, 0xc2, 
    0x5, 0x8, 0x5, 0x2, 0xbe, 0xc2, 0x5, 0xa, 0x6, 0x2, 0xbf, 0xc2, 0x5, 
    0xc, 0x7, 0x2, 0xc0, 0xc2, 0x5, 0xe, 0x8, 0x2, 0xc1, 0xbd, 0x3, 0x2, 
    0x2, 0x2, 0xc1, 0xbe, 0x3, 0x2, 0x2, 0x2, 0xc1, 0xbf, 0x3, 0x2, 0x2, 
    0x2, 0xc1, 0xc0, 0x3, 0x2, 0x2, 0x2, 0xc2, 0x7, 0x3, 0x2, 0x2, 0x2, 
    0xc3, 0xc4, 0x7, 0x32, 0x2, 0x2, 0xc4, 0xc5, 0x7, 0x38, 0x2, 0x2, 0xc5, 
    0xc6, 0x7, 0x8, 0x2, 0x2, 0xc6, 0x9, 0x3, 0x2, 0x2, 0x2, 0xc7, 0xc8, 
    0x7, 0x32, 0x2, 0x2, 0xc8, 0xcb, 0x7, 0x3a, 0x2, 0x2, 0xc9, 0xca, 0x9, 
    0x2, 0x2, 0x2, 0xca, 0xcc, 0x5, 0x7a, 0x3e, 0x2, 0xcb, 0xc9, 0x3, 0x2, 
    0x2, 0x2, 0xcb, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xcc, 0xcd, 0x3, 0x2, 0x2, 
    0x2, 0xcd, 0xce, 0x7, 0x8, 0x2, 0x2, 0xce, 0xb, 0x3, 0x2, 0x2, 0x2, 
    0xcf, 0xd0, 0x7, 0x32, 0x2, 0x2, 0xd0, 0xd1, 0x7, 0x3c, 0x2, 0x2, 0xd1, 
    0xd2, 0x9, 0x2, 0x2, 0x2, 0xd2, 0xd5, 0x5, 0x76, 0x3c, 0x2, 0xd3, 0xd4, 
    0x9, 0x2, 0x2, 0x2, 0xd4, 0xd6, 0x5, 0x7a, 0x3e, 0x2, 0xd5, 0xd3, 0x3, 
    0x2, 0x2, 0x2, 0xd5, 0xd6, 0x3, 0x2, 0x2, 0x2, 0xd6, 0xd7, 0x3, 0x2, 
    0x2, 0x2, 0xd7, 0xd8, 0x7, 0x8, 0x2, 0x2, 0xd8, 0xd, 0x3, 0x2, 0x2, 
    0x2, 0xd9, 0xda, 0x7, 0x32, 0x2, 0x2, 0xda, 0xdb, 0x7, 0x3e, 0x2, 0x2, 
    0xdb, 0xdc, 0x9, 0x2, 0x2, 0x2, 0xdc, 0xdf, 0x5, 0x76, 0x3c, 0x2, 0xdd, 
    0xde, 0x9, 0x2, 0x2, 0x2, 0xde, 0xe0, 0x5, 0x7a, 0x3e, 0x2, 0xdf, 0xdd, 
    0x3, 0x2, 0x2, 0x2, 0xdf, 0xe0, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xe1, 0x3, 
    0x2, 0x2, 0x2, 0xe1, 0xe2, 0x7, 0x8, 0x2, 0x2, 0xe2, 0xf, 0x3, 0x2, 
    0x2, 0x2, 0xe3, 0xe4, 0x7, 0x32, 0x2, 0x2, 0xe4, 0xe5, 0x7, 0x52, 0x2, 
    0x2, 0xe5, 0xe6, 0x7, 0x3b, 0x2, 0x2, 0xe6, 0xe7, 0x7, 0x25, 0x2, 0x2, 
    0xe7, 0xe8, 0x5, 0x12, 0xa, 0x2, 0xe8, 0x11, 0x3, 0x2, 0x2, 0x2, 0xe9, 
    0xea, 0x7, 0x40, 0x2, 0x2, 0xea, 0xeb, 0x5, 0x4a, 0x26, 0x2, 0xeb, 0xec, 
    0x7, 0x41, 0x2, 0x2, 0xec, 0xee, 0x5, 0x62, 0x32, 0x2, 0xed, 0xef, 0x5, 
    0x64, 0x33, 0x2, 0xee, 0xed, 0x3, 0x2, 0x2, 0x2, 0xee, 0xef, 0x3, 0x2, 
    0x2, 0x2, 0xef, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xf1, 0x7, 0x43, 0x2, 
    0x2, 0xf1, 0xf3, 0x5, 0x50, 0x29, 0x2, 0xf2, 0xf0, 0x3, 0x2, 0x2, 0x2, 
    0xf2, 0xf3, 0x3, 0x2, 0x2, 0x2, 0xf3, 0xf7, 0x3, 0x2, 0x2, 0x2, 0xf4, 
    0xf5, 0x7, 0x44, 0x2, 0x2, 0xf5, 0xf6, 0x7, 0x4b, 0x2, 0x2, 0xf6, 0xf8, 
    0x5, 0x5e, 0x30, 0x2, 0xf7, 0xf4, 0x3, 0x2, 0x2, 0x2, 0xf7, 0xf8, 0x3, 
    0x2, 0x2, 0x2, 0xf8, 0xfc, 0x3, 0x2, 0x2, 0x2, 0xf9, 0xfa, 0x7, 0x4e, 
    0x2, 0x2, 0xfa, 0xfb, 0x7, 0x4b, 0x2, 0x2, 0xfb, 0xfd, 0x5, 0x52, 0x2a, 
    0x2, 0xfc, 0xf9, 0x3, 0x2, 0x2, 0x2, 0xfc, 0xfd, 0x3, 0x2, 0x2, 0x2, 
    0xfd, 0x100, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xff, 0x7, 0x50, 0x2, 0x2, 0xff, 
    0x101, 0x5, 0x82, 0x42, 0x2, 0x100, 0xfe, 0x3, 0x2, 0x2, 0x2, 0x100, 
    0x101, 0x3, 0x2, 0x2, 0x2, 0x101, 0x104, 0x3, 0x2, 0x2, 0x2, 0x102, 
    0x103, 0x7, 0x51, 0x2, 0x2, 0x103, 0x105, 0x5, 0x84, 0x43, 0x2, 0x104, 
    0x102, 0x3, 0x2, 0x2, 0x2, 0x104, 0x105, 0x3, 0x2, 0x2, 0x2, 0x105, 
    0x106, 0x3, 0x2, 0x2, 0x2, 0x106, 0x107, 0x7, 0x8, 0x2, 0x2, 0x107, 
    0x13, 0x3, 0x2, 0x2, 0x2, 0x108, 0x109, 0x7, 0x2a, 0x2, 0x2, 0x109, 
    0x10a, 0x7, 0x37, 0x2, 0x2, 0x10a, 0x10c, 0x5, 0x7a, 0x3e, 0x2, 0x10b, 
    0x10d, 0x5, 0x86, 0x44, 0x2, 0x10c, 0x10b, 0x3, 0x2, 0x2, 0x2, 0x10c, 
    0x10d, 0x3, 0x2, 0x2, 0x2, 0x10d, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x10e, 
    0x10f, 0x7, 0x8, 0x2, 0x2, 0x10f, 0x15, 0x3, 0x2, 0x2, 0x2, 0x110, 0x111, 
    0x7, 0x2c, 0x2, 0x2, 0x111, 0x112, 0x7, 0x37, 0x2, 0x2, 0x112, 0x113, 
    0x5, 0x7a, 0x3e, 0x2, 0x113, 0x114, 0x7, 0x8, 0x2, 0x2, 0x114, 0x17, 
    0x3, 0x2, 0x2, 0x2, 0x115, 0x116, 0x7, 0x2a, 0x2, 0x2, 0x116, 0x117, 
    0x7, 0x39, 0x2, 0x2, 0x117, 0x119, 0x5, 0x76, 0x3c, 0x2, 0x118, 0x11a, 
    0x5, 0x86, 0x44, 0x2, 0x119, 0x118, 0x3, 0x2, 0x2, 0x2, 0x119, 0x11a, 
    0x3, 0x2, 0x2, 0x2, 0x11a, 0x11b, 0x3, 0x2, 0x2, 0x2, 0x11b, 0x11c, 
    0x7, 0x91, 0x2, 0x2, 0x11c, 0x11d, 0x5, 0x24, 0x13, 0x2, 0x11d, 0x11e, 
    0x7, 0x92, 0x2, 0x2, 0x11e, 0x11f, 0x7, 0x8, 0x2, 0x2, 0x11f, 0x19, 
    0x3, 0x2, 0x2, 0x2, 0x120, 0x121, 0x7, 0x2c, 0x2, 0x2, 0x121, 0x122, 
    0x7, 0x39, 0x2, 0x2, 0x122, 0x123, 0x5, 0x76, 0x3c, 0x2, 0x123, 0x124, 
    0x7, 0x8, 0x2, 0x2, 0x124, 0x1b, 0x3, 0x2, 0x2, 0x2, 0x125, 0x126, 0x7, 
    0x2d, 0x2, 0x2, 0x126, 0x127, 0x7, 0x39, 0x2, 0x2, 0x127, 0x128, 0x5, 
    0x76, 0x3c, 0x2, 0x128, 0x129, 0x5, 0x2e, 0x18, 0x2, 0x129, 0x12a, 0x7, 
    0x8, 0x2, 0x2, 0x12a, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x12c, 0x7, 0x2d, 
    0x2, 0x2, 0x12c, 0x12d, 0x7, 0x37, 0x2, 0x2, 0x12d, 0x12e, 0x5, 0x7a, 
    0x3e, 0x2, 0x12e, 0x12f, 0x5, 0x28, 0x15, 0x2, 0x12f, 0x130, 0x7, 0x8, 
    0x2, 0x2, 0x130, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x131, 0x132, 0x7, 0x2a, 
    0x2, 0x2, 0x132, 0x133, 0x7, 0x26, 0x2, 0x2, 0x133, 0x134, 0x5, 0x7e, 
    0x40, 0x2, 0x134, 0x135, 0x7, 0x4d, 0x2, 0x2, 0x135, 0x136, 0x5, 0x76, 
    0x3c, 0x2, 0x136, 0x137, 0x7, 0x91, 0x2, 0x2, 0x137, 0x138, 0x5, 0x5a, 
    0x2e, 0x2, 0x138, 0x139, 0x7, 0x92, 0x2, 0x2, 0x139, 0x13a, 0x7, 0x8, 
    0x2, 0x2, 0x13a, 0x21, 0x3, 0x2, 0x2, 0x2, 0x13b, 0x13c, 0x7, 0x30, 
    0x2, 0x2, 0x13c, 0x13d, 0x7, 0x31, 0x2, 0x2, 0x13d, 0x13e, 0x5, 0x76, 
    0x3c, 0x2, 0x13e, 0x13f, 0x7, 0x91, 0x2, 0x2, 0x13f, 0x140, 0x5, 0x58, 
    0x2d, 0x2, 0x140, 0x141, 0x7, 0x92, 0x2, 0x2, 0x141, 0x142, 0x7, 0x3f, 
    0x2, 0x2, 0x142, 0x143, 0x7, 0x91, 0x2, 0x2, 0x143, 0x144, 0x5, 0x56, 
    0x2c, 0x2, 0x144, 0x145, 0x7, 0x92, 0x2, 0x2, 0x145, 0x146, 0x7, 0x8, 
    0x2, 0x2, 0x146, 0x23, 0x3, 0x2, 0x2, 0x2, 0x147, 0x14c, 0x5, 0x26, 
    0x14, 0x2, 0x148, 0x149, 0x7, 0xd, 0x2, 0x2, 0x149, 0x14b, 0x5, 0x26, 
    0x14, 0x2, 0x14a, 0x148, 0x3, 0x2, 0x2, 0x2, 0x14b, 0x14e, 0x3, 0x2, 
    0x2, 0x2, 0x14c, 0x14a, 0x3, 0x2, 0x2, 0x2, 0x14c, 0x14d, 0x3, 0x2, 
    0x2, 0x2, 0x14d, 0x25, 0x3, 0x2, 0x2, 0x2, 0x14e, 0x14c, 0x3, 0x2, 0x2, 
    0x2, 0x14f, 0x152, 0x5, 0x46, 0x24, 0x2, 0x150, 0x152, 0x5, 0x48, 0x25, 
    0x2, 0x151, 0x14f, 0x3, 0x2, 0x2, 0x2, 0x151, 0x150, 0x3, 0x2, 0x2, 
    0x2, 0x152, 0x27, 0x3, 0x2, 0x2, 0x2, 0x153, 0x158, 0x5, 0x2a, 0x16, 
    0x2, 0x154, 0x155, 0x7, 0xd, 0x2, 0x2, 0x155, 0x157, 0x5, 0x2a, 0x16, 
    0x2, 0x156, 0x154, 0x3, 0x2, 0x2, 0x2, 0x157, 0x15a, 0x3, 0x2, 0x2, 
    0x2, 0x158, 0x156, 0x3, 0x2, 0x2, 0x2, 0x158, 0x159, 0x3, 0x2, 0x2, 
    0x2, 0x159, 0x29, 0x3, 0x2, 0x2, 0x2, 0x15a, 0x158, 0x3, 0x2, 0x2, 0x2, 
    0x15b, 0x15e, 0x5, 0x2c, 0x17, 0x2, 0x15c, 0x15e, 0x5, 0x40, 0x21, 0x2, 
    0x15d, 0x15b, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15c, 0x3, 0x2, 0x2, 0x2, 
    0x15e, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x15f, 0x160, 0x7, 0x2e, 0x2, 0x2, 
    0x160, 0x161, 0x7, 0x47, 0x2, 0x2, 0x161, 0x162, 0x5, 0x7a, 0x3e, 0x2, 
    0x162, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x163, 0x168, 0x5, 0x30, 0x19, 0x2, 
    0x164, 0x165, 0x7, 0xd, 0x2, 0x2, 0x165, 0x167, 0x5, 0x30, 0x19, 0x2, 
    0x166, 0x164, 0x3, 0x2, 0x2, 0x2, 0x167, 0x16a, 0x3, 0x2, 0x2, 0x2, 
    0x168, 0x166, 0x3, 0x2, 0x2, 0x2, 0x168, 0x169, 0x3, 0x2, 0x2, 0x2, 
    0x169, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x16a, 0x168, 0x3, 0x2, 0x2, 0x2, 0x16b, 
    0x174, 0x5, 0x32, 0x1a, 0x2, 0x16c, 0x174, 0x5, 0x34, 0x1b, 0x2, 0x16d, 
    0x174, 0x5, 0x36, 0x1c, 0x2, 0x16e, 0x174, 0x5, 0x38, 0x1d, 0x2, 0x16f, 
    0x174, 0x5, 0x3a, 0x1e, 0x2, 0x170, 0x174, 0x5, 0x3c, 0x1f, 0x2, 0x171, 
    0x174, 0x5, 0x3e, 0x20, 0x2, 0x172, 0x174, 0x5, 0x40, 0x21, 0x2, 0x173, 
    0x16b, 0x3, 0x2, 0x2, 0x2, 0x173, 0x16c, 0x3, 0x2, 0x2, 0x2, 0x173, 
    0x16d, 0x3, 0x2, 0x2, 0x2, 0x173, 0x16e, 0x3, 0x2, 0x2, 0x2, 0x173, 
    0x16f, 0x3, 0x2, 0x2, 0x2, 0x173, 0x170, 0x3, 0x2, 0x2, 0x2, 0x173, 
    0x171, 0x3, 0x2, 0x2, 0x2, 0x173, 0x172, 0x3, 0x2, 0x2, 0x2, 0x174, 
    0x31, 0x3, 0x2, 0x2, 0x2, 0x175, 0x176, 0x7, 0x2b, 0x2, 0x2, 0x176, 
    0x177, 0x5, 0x78, 0x3d, 0x2, 0x177, 0x178, 0x5, 0x90, 0x49, 0x2, 0x178, 
    0x33, 0x3, 0x2, 0x2, 0x2, 0x179, 0x17a, 0x7, 0x2c, 0x2, 0x2, 0x17a, 
    0x17b, 0x7, 0x3b, 0x2, 0x2, 0x17b, 0x17c, 0x5, 0x78, 0x3d, 0x2, 0x17c, 
    0x35, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x17e, 0x7, 0x2d, 0x2, 0x2, 0x17e, 
    0x17f, 0x7, 0x3b, 0x2, 0x2, 0x17f, 0x180, 0x5, 0x78, 0x3d, 0x2, 0x180, 
    0x181, 0x5, 0x90, 0x49, 0x2, 0x181, 0x37, 0x3, 0x2, 0x2, 0x2, 0x182, 
    0x183, 0x7, 0x2e, 0x2, 0x2, 0x183, 0x184, 0x7, 0x3b, 0x2, 0x2, 0x184, 
    0x185, 0x5, 0x42, 0x22, 0x2, 0x185, 0x186, 0x7, 0x47, 0x2, 0x2, 0x186, 
    0x187, 0x5, 0x44, 0x23, 0x2, 0x187, 0x39, 0x3, 0x2, 0x2, 0x2, 0x188, 
    0x189, 0x7, 0x2e, 0x2, 0x2, 0x189, 0x18a, 0x7, 0x47, 0x2, 0x2, 0x18a, 
    0x18b, 0x5, 0x76, 0x3c, 0x2, 0x18b, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x18c, 
    0x18d, 0x7, 0x2b, 0x2, 0x2, 0x18d, 0x18e, 0x5, 0x8a, 0x46, 0x2, 0x18e, 
    0x18f, 0x5, 0x80, 0x41, 0x2, 0x18f, 0x190, 0x7, 0x91, 0x2, 0x2, 0x190, 
    0x191, 0x5, 0x5c, 0x2f, 0x2, 0x191, 0x192, 0x7, 0x92, 0x2, 0x2, 0x192, 
    0x3d, 0x3, 0x2, 0x2, 0x2, 0x193, 0x194, 0x7, 0x2c, 0x2, 0x2, 0x194, 
    0x195, 0x5, 0x8a, 0x46, 0x2, 0x195, 0x196, 0x5, 0x80, 0x41, 0x2, 0x196, 
    0x3f, 0x3, 0x2, 0x2, 0x2, 0x197, 0x198, 0x7, 0x2d, 0x2, 0x2, 0x198, 
    0x199, 0x7, 0x3d, 0x2, 0x2, 0x199, 0x19a, 0x7, 0x33, 0x2, 0x2, 0x19a, 
    0x19b, 0x5, 0x86, 0x44, 0x2, 0x19b, 0x41, 0x3, 0x2, 0x2, 0x2, 0x19c, 
    0x19d, 0x5, 0x78, 0x3d, 0x2, 0x19d, 0x43, 0x3, 0x2, 0x2, 0x2, 0x19e, 
    0x19f, 0x5, 0x78, 0x3d, 0x2, 0x19f, 0x45, 0x3, 0x2, 0x2, 0x2, 0x1a0, 
    0x1a1, 0x5, 0x78, 0x3d, 0x2, 0x1a1, 0x1a3, 0x5, 0x90, 0x49, 0x2, 0x1a2, 
    0x1a4, 0x5, 0x8a, 0x46, 0x2, 0x1a3, 0x1a2, 0x3, 0x2, 0x2, 0x2, 0x1a3, 
    0x1a4, 0x3, 0x2, 0x2, 0x2, 0x1a4, 0x47, 0x3, 0x2, 0x2, 0x2, 0x1a5, 0x1a6, 
    0x5, 0x8a, 0x46, 0x2, 0x1a6, 0x1a7, 0x5, 0x80, 0x41, 0x2, 0x1a7, 0x1a8, 
    0x7, 0x91, 0x2, 0x2, 0x1a8, 0x1a9, 0x5, 0x5c, 0x2f, 0x2, 0x1a9, 0x1aa, 
    0x7, 0x92, 0x2, 0x2, 0x1aa, 0x49, 0x3, 0x2, 0x2, 0x2, 0x1ab, 0x1ae, 
    0x5, 0x4c, 0x27, 0x2, 0x1ac, 0x1ae, 0x5, 0x4e, 0x28, 0x2, 0x1ad, 0x1ab, 
    0x3, 0x2, 0x2, 0x2, 0x1ad, 0x1ac, 0x3, 0x2, 0x2, 0x2, 0x1ae, 0x1b6, 
    0x3, 0x2, 0x2, 0x2, 0x1af, 0x1b2, 0x7, 0xd, 0x2, 0x2, 0x1b0, 0x1b3, 
    0x5, 0x4c, 0x27, 0x2, 0x1b1, 0x1b3, 0x5, 0x4e, 0x28, 0x2, 0x1b2, 0x1b0, 
    0x3, 0x2, 0x2, 0x2, 0x1b2, 0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b3, 0x1b5, 
    0x3, 0x2, 0x2, 0x2, 0x1b4, 0x1af, 0x3, 0x2, 0x2, 0x2, 0x1b5, 0x1b8, 
    0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b4, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b7, 
    0x3, 0x2, 0x2, 0x2, 0x1b7, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1b6, 0x3, 
    0x2, 0x2, 0x2, 0x1b9, 0x1bc, 0x5, 0x8e, 0x48, 0x2, 0x1ba, 0x1bb, 0x7, 
    0x45, 0x2, 0x2, 0x1bb, 0x1bd, 0x5, 0x7c, 0x3f, 0x2, 0x1bc, 0x1ba, 0x3, 
    0x2, 0x2, 0x2, 0x1bc, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1c0, 0x3, 
    0x2, 0x2, 0x2, 0x1be, 0x1bf, 0x7, 0x80, 0x2, 0x2, 0x1bf, 0x1c1, 0x5, 
    0x8c, 0x47, 0x2, 0x1c0, 0x1be, 0x3, 0x2, 0x2, 0x2, 0x1c0, 0x1c1, 0x3, 
    0x2, 0x2, 0x2, 0x1c1, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x1c2, 0x1c3, 0x7, 0x8a, 
    0x2, 0x2, 0x1c3, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x1c4, 0x1c5, 0x5, 0x8e, 
    0x48, 0x2, 0x1c5, 0x51, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1cb, 0x5, 0x54, 
    0x2b, 0x2, 0x1c7, 0x1c8, 0x7, 0xd, 0x2, 0x2, 0x1c8, 0x1ca, 0x5, 0x54, 
    0x2b, 0x2, 0x1c9, 0x1c7, 0x3, 0x2, 0x2, 0x2, 0x1ca, 0x1cd, 0x3, 0x2, 
    0x2, 0x2, 0x1cb, 0x1c9, 0x3, 0x2, 0x2, 0x2, 0x1cb, 0x1cc, 0x3, 0x2, 
    0x2, 0x2, 0x1cc, 0x53, 0x3, 0x2, 0x2, 0x2, 0x1cd, 0x1cb, 0x3, 0x2, 0x2, 
    0x2, 0x1ce, 0x1d0, 0x5, 0x8e, 0x48, 0x2, 0x1cf, 0x1d1, 0x7, 0x4f, 0x2, 
    0x2, 0x1d0, 0x1cf, 0x3, 0x2, 0x2, 0x2, 0x1d0, 0x1d1, 0x3, 0x2, 0x2, 
    0x2, 0x1d1, 0x55, 0x3, 0x2, 0x2, 0x2, 0x1d2, 0x1d7, 0x5, 0x88, 0x45, 
    0x2, 0x1d3, 0x1d4, 0x7, 0xd, 0x2, 0x2, 0x1d4, 0x1d6, 0x5, 0x88, 0x45, 
    0x2, 0x1d5, 0x1d3, 0x3, 0x2, 0x2, 0x2, 0x1d6, 0x1d9, 0x3, 0x2, 0x2, 
    0x2, 0x1d7, 0x1d5, 0x3, 0x2, 0x2, 0x2, 0x1d7, 0x1d8, 0x3, 0x2, 0x2, 
    0x2, 0x1d8, 0x57, 0x3, 0x2, 0x2, 0x2, 0x1d9, 0x1d7, 0x3, 0x2, 0x2, 0x2, 
    0x1da, 0x1df, 0x5, 0x74, 0x3b, 0x2, 0x1db, 0x1dc, 0x7, 0xd, 0x2, 0x2, 
    0x1dc, 0x1de, 0x5, 0x74, 0x3b, 0x2, 0x1dd, 0x1db, 0x3, 0x2, 0x2, 0x2, 
    0x1de, 0x1e1, 0x3, 0x2, 0x2, 0x2, 0x1df, 0x1dd, 0x3, 0x2, 0x2, 0x2, 
    0x1df, 0x1e0, 0x3, 0x2, 0x2, 0x2, 0x1e0, 0x59, 0x3, 0x2, 0x2, 0x2, 0x1e1, 
    0x1df, 0x3, 0x2, 0x2, 0x2, 0x1e2, 0x1e7, 0x5, 0x78, 0x3d, 0x2, 0x1e3, 
    0x1e4, 0x7, 0xd, 0x2, 0x2, 0x1e4, 0x1e6, 0x5, 0x78, 0x3d, 0x2, 0x1e5, 
    0x1e3, 0x3, 0x2, 0x2, 0x2, 0x1e6, 0x1e9, 0x3, 0x2, 0x2, 0x2, 0x1e7, 
    0x1e5, 0x3, 0x2, 0x2, 0x2, 0x1e7, 0x1e8, 0x3, 0x2, 0x2, 0x2, 0x1e8, 
    0x5b, 0x3, 0x2, 0x2, 0x2, 0x1e9, 0x1e7, 0x3, 0x2, 0x2, 0x2, 0x1ea, 0x1ef, 
    0x5, 0x78, 0x3d, 0x2, 0x1eb, 0x1ec, 0x7, 0xd, 0x2, 0x2, 0x1ec, 0x1ee, 
    0x5, 0x78, 0x3d, 0x2, 0x1ed, 0x1eb, 0x3, 0x2, 0x2, 0x2, 0x1ee, 0x1f1, 
    0x3, 0x2, 0x2, 0x2, 0x1ef, 0x1ed, 0x3, 0x2, 0x2, 0x2, 0x1ef, 0x1f0, 
    0x3, 0x2, 0x2, 0x2, 0x1f0, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x1f1, 0x1ef, 0x3, 
    0x2, 0x2, 0x2, 0x1f2, 0x1f7, 0x5, 0x60, 0x31, 0x2, 0x1f3, 0x1f4, 0x7, 
    0xd, 0x2, 0x2, 0x1f4, 0x1f6, 0x5, 0x60, 0x31, 0x2, 0x1f5, 0x1f3, 0x3, 
    0x2, 0x2, 0x2, 0x1f6, 0x1f9, 0x3, 0x2, 0x2, 0x2, 0x1f7, 0x1f5, 0x3, 
    0x2, 0x2, 0x2, 0x1f7, 0x1f8, 0x3, 0x2, 0x2, 0x2, 0x1f8, 0x5f, 0x3, 0x2, 
    0x2, 0x2, 0x1f9, 0x1f7, 0x3, 0x2, 0x2, 0x2, 0x1fa, 0x1fb, 0x5, 0x8e, 
    0x48, 0x2, 0x1fb, 0x61, 0x3, 0x2, 0x2, 0x2, 0x1fc, 0x201, 0x5, 0x72, 
    0x3a, 0x2, 0x1fd, 0x1fe, 0x7, 0xd, 0x2, 0x2, 0x1fe, 0x200, 0x5, 0x72, 
    0x3a, 0x2, 0x1ff, 0x1fd, 0x3, 0x2, 0x2, 0x2, 0x200, 0x203, 0x3, 0x2, 
    0x2, 0x2, 0x201, 0x1ff, 0x3, 0x2, 0x2, 0x2, 0x201, 0x202, 0x3, 0x2, 
    0x2, 0x2, 0x202, 0x63, 0x3, 0x2, 0x2, 0x2, 0x203, 0x201, 0x3, 0x2, 0x2, 
    0x2, 0x204, 0x206, 0x5, 0x66, 0x34, 0x2, 0x205, 0x204, 0x3, 0x2, 0x2, 
    0x2, 0x206, 0x207, 0x3, 0x2, 0x2, 0x2, 0x207, 0x205, 0x3, 0x2, 0x2, 
    0x2, 0x207, 0x208, 0x3, 0x2, 0x2, 0x2, 0x208, 0x65, 0x3, 0x2, 0x2, 0x2, 
    0x209, 0x20b, 0x5, 0x70, 0x39, 0x2, 0x20a, 0x209, 0x3, 0x2, 0x2, 0x2, 
    0x20a, 0x20b, 0x3, 0x2, 0x2, 0x2, 0x20b, 0x20c, 0x3, 0x2, 0x2, 0x2, 
    0x20c, 0x20d, 0x7, 0x42, 0x2, 0x2, 0x20d, 0x20e, 0x5, 0x68, 0x35, 0x2, 
    0x20e, 0x20f, 0x7, 0x4d, 0x2, 0x2, 0x20f, 0x210, 0x5, 0x6a, 0x36, 0x2, 
    0x210, 0x211, 0x5, 0x6e, 0x38, 0x2, 0x211, 0x212, 0x5, 0x6c, 0x37, 0x2, 
    0x212, 0x67, 0x3, 0x2, 0x2, 0x2, 0x213, 0x216, 0x5, 0x76, 0x3c, 0x2, 
    0x214, 0x215, 0x7, 0x45, 0x2, 0x2, 0x215, 0x217, 0x5, 0x7c, 0x3f, 0x2, 
    0x216, 0x214, 0x3, 0x2, 0x2, 0x2, 0x216, 0x217, 0x3, 0x2, 0x2, 0x2, 
    0x217, 0x69, 0x3, 0x2, 0x2, 0x2, 0x218, 0x219, 0x5, 0x74, 0x3b, 0x2, 
    0x219, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x21a, 0x21b, 0x5, 0x74, 0x3b, 0x2, 
    0x21b, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x21c, 0x21d, 0x9, 0x3, 0x2, 0x2, 0x21d, 
    0x6f, 0x3, 0x2, 0x2, 0x2, 0x21e, 0x224, 0x7, 0x36, 0x2, 0x2, 0x21f, 
    0x224, 0x7, 0x7c, 0x2, 0x2, 0x220, 0x224, 0x7, 0x7d, 0x2, 0x2, 0x221, 
    0x222, 0x7, 0x34, 0x2, 0x2, 0x222, 0x224, 0x7, 0x35, 0x2, 0x2, 0x223, 
    0x21e, 0x3, 0x2, 0x2, 0x2, 0x223, 0x21f, 0x3, 0x2, 0x2, 0x2, 0x223, 
    0x220, 0x3, 0x2, 0x2, 0x2, 0x223, 0x221, 0x3, 0x2, 0x2, 0x2, 0x224, 
    0x71, 0x3, 0x2, 0x2, 0x2, 0x225, 0x228, 0x5, 0x76, 0x3c, 0x2, 0x226, 
    0x227, 0x7, 0x45, 0x2, 0x2, 0x227, 0x229, 0x5, 0x7c, 0x3f, 0x2, 0x228, 
    0x226, 0x3, 0x2, 0x2, 0x2, 0x228, 0x229, 0x3, 0x2, 0x2, 0x2, 0x229, 
    0x73, 0x3, 0x2, 0x2, 0x2, 0x22a, 0x230, 0x5, 0x78, 0x3d, 0x2, 0x22b, 
    0x22c, 0x5, 0x76, 0x3c, 0x2, 0x22c, 0x22d, 0x7, 0xe, 0x2, 0x2, 0x22d, 
    0x22e, 0x5, 0x78, 0x3d, 0x2, 0x22e, 0x230, 0x3, 0x2, 0x2, 0x2, 0x22f, 
    0x22a, 0x3, 0x2, 0x2, 0x2, 0x22f, 0x22b, 0x3, 0x2, 0x2, 0x2, 0x230, 
    0x75, 0x3, 0x2, 0x2, 0x2, 0x231, 0x232, 0x9, 0x4, 0x2, 0x2, 0x232, 0x77, 
    0x3, 0x2, 0x2, 0x2, 0x233, 0x234, 0x9, 0x4, 0x2, 0x2, 0x234, 0x79, 0x3, 
    0x2, 0x2, 0x2, 0x235, 0x236, 0x9, 0x4, 0x2, 0x2, 0x236, 0x7b, 0x3, 0x2, 
    0x2, 0x2, 0x237, 0x238, 0x9, 0x4, 0x2, 0x2, 0x238, 0x7d, 0x3, 0x2, 0x2, 
    0x2, 0x239, 0x23a, 0x9, 0x4, 0x2, 0x2, 0x23a, 0x7f, 0x3, 0x2, 0x2, 0x2, 
    0x23b, 0x23c, 0x9, 0x4, 0x2, 0x2, 0x23c, 0x81, 0x3, 0x2, 0x2, 0x2, 0x23d, 
    0x23e, 0x7, 0xa2, 0x2, 0x2, 0x23e, 0x83, 0x3, 0x2, 0x2, 0x2, 0x23f, 
    0x240, 0x7, 0xa2, 0x2, 0x2, 0x240, 0x85, 0x3, 0x2, 0x2, 0x2, 0x241, 
    0x242, 0x7, 0xa2, 0x2, 0x2, 0x242, 0x87, 0x3, 0x2, 0x2, 0x2, 0x243, 
    0x245, 0x7, 0x89, 0x2, 0x2, 0x244, 0x243, 0x3, 0x2, 0x2, 0x2, 0x244, 
    0x245, 0x3, 0x2, 0x2, 0x2, 0x245, 0x246, 0x3, 0x2, 0x2, 0x2, 0x246, 
    0x251, 0x7, 0xa2, 0x2, 0x2, 0x247, 0x249, 0x7, 0x89, 0x2, 0x2, 0x248, 
    0x247, 0x3, 0x2, 0x2, 0x2, 0x248, 0x249, 0x3, 0x2, 0x2, 0x2, 0x249, 
    0x24a, 0x3, 0x2, 0x2, 0x2, 0x24a, 0x251, 0x7, 0xa1, 0x2, 0x2, 0x24b, 
    0x251, 0x5, 0x92, 0x4a, 0x2, 0x24c, 0x251, 0x7, 0x4a, 0x2, 0x2, 0x24d, 
    0x251, 0x7, 0x11, 0x2, 0x2, 0x24e, 0x251, 0x7, 0x3, 0x2, 0x2, 0x24f, 
    0x251, 0x7, 0x9e, 0x2, 0x2, 0x250, 0x244, 0x3, 0x2, 0x2, 0x2, 0x250, 
    0x248, 0x3, 0x2, 0x2, 0x2, 0x250, 0x24b, 0x3, 0x2, 0x2, 0x2, 0x250, 
    0x24c, 0x3, 0x2, 0x2, 0x2, 0x250, 0x24d, 0x3, 0x2, 0x2, 0x2, 0x250, 
    0x24e, 0x3, 0x2, 0x2, 0x2, 0x250, 0x24f, 0x3, 0x2, 0x2, 0x2, 0x251, 
    0x89, 0x3, 0x2, 0x2, 0x2, 0x252, 0x257, 0x7, 0x27, 0x2, 0x2, 0x253, 
    0x257, 0x7, 0x26, 0x2, 0x2, 0x254, 0x255, 0x7, 0x49, 0x2, 0x2, 0x255, 
    0x257, 0x7, 0x4a, 0x2, 0x2, 0x256, 0x252, 0x3, 0x2, 0x2, 0x2, 0x256, 
    0x253, 0x3, 0x2, 0x2, 0x2, 0x256, 0x254, 0x3, 0x2, 0x2, 0x2, 0x257, 
    0x8b, 0x3, 0x2, 0x2, 0x2, 0x258, 0x259, 0x5, 0x90, 0x49, 0x2, 0x259, 
    0x8d, 0x3, 0x2, 0x2, 0x2, 0x25a, 0x25b, 0x8, 0x48, 0x1, 0x2, 0x25b, 
    0x25c, 0x7, 0x97, 0x2, 0x2, 0x25c, 0x3a1, 0x5, 0x8e, 0x48, 0x54, 0x25d, 
    0x25e, 0x7, 0x89, 0x2, 0x2, 0x25e, 0x3a1, 0x5, 0x8e, 0x48, 0x53, 0x25f, 
    0x260, 0x7, 0x62, 0x2, 0x2, 0x260, 0x261, 0x7, 0x91, 0x2, 0x2, 0x261, 
    0x262, 0x5, 0x8e, 0x48, 0x2, 0x262, 0x263, 0x7, 0x92, 0x2, 0x2, 0x263, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x264, 0x265, 0x7, 0x63, 0x2, 0x2, 0x265, 
    0x266, 0x7, 0x91, 0x2, 0x2, 0x266, 0x267, 0x5, 0x8e, 0x48, 0x2, 0x267, 
    0x268, 0x7, 0x92, 0x2, 0x2, 0x268, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x269, 
    0x26a, 0x7, 0x64, 0x2, 0x2, 0x26a, 0x26b, 0x7, 0x91, 0x2, 0x2, 0x26b, 
    0x26c, 0x5, 0x8e, 0x48, 0x2, 0x26c, 0x26d, 0x7, 0x92, 0x2, 0x2, 0x26d, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x26e, 0x26f, 0x7, 0x65, 0x2, 0x2, 0x26f, 
    0x270, 0x7, 0x91, 0x2, 0x2, 0x270, 0x271, 0x5, 0x8e, 0x48, 0x2, 0x271, 
    0x272, 0x7, 0x92, 0x2, 0x2, 0x272, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x273, 
    0x274, 0x7, 0x66, 0x2, 0x2, 0x274, 0x275, 0x7, 0x91, 0x2, 0x2, 0x275, 
    0x276, 0x5, 0x8e, 0x48, 0x2, 0x276, 0x277, 0x7, 0x92, 0x2, 0x2, 0x277, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x278, 0x279, 0x7, 0x67, 0x2, 0x2, 0x279, 
    0x27a, 0x7, 0x91, 0x2, 0x2, 0x27a, 0x27b, 0x5, 0x8e, 0x48, 0x2, 0x27b, 
    0x27c, 0x7, 0x92, 0x2, 0x2, 0x27c, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x27d, 
    0x27e, 0x7, 0x68, 0x2, 0x2, 0x27e, 0x27f, 0x7, 0x91, 0x2, 0x2, 0x27f, 
    0x280, 0x5, 0x8e, 0x48, 0x2, 0x280, 0x281, 0x7, 0x92, 0x2, 0x2, 0x281, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x282, 0x283, 0x7, 0x69, 0x2, 0x2, 0x283, 
    0x284, 0x7, 0x91, 0x2, 0x2, 0x284, 0x285, 0x5, 0x8e, 0x48, 0x2, 0x285, 
    0x286, 0x7, 0x92, 0x2, 0x2, 0x286, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x287, 
    0x288, 0x7, 0x6b, 0x2, 0x2, 0x288, 0x289, 0x7, 0x91, 0x2, 0x2, 0x289, 
    0x28a, 0x5, 0x8e, 0x48, 0x2, 0x28a, 0x28b, 0x7, 0x92, 0x2, 0x2, 0x28b, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x28c, 0x28d, 0x7, 0x6c, 0x2, 0x2, 0x28d, 
    0x28e, 0x7, 0x91, 0x2, 0x2, 0x28e, 0x28f, 0x5, 0x8e, 0x48, 0x2, 0x28f, 
    0x290, 0x7, 0x92, 0x2, 0x2, 0x290, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x291, 
    0x292, 0x7, 0x6d, 0x2, 0x2, 0x292, 0x293, 0x7, 0x91, 0x2, 0x2, 0x293, 
    0x294, 0x5, 0x8e, 0x48, 0x2, 0x294, 0x295, 0x7, 0x92, 0x2, 0x2, 0x295, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x296, 0x297, 0x7, 0x6f, 0x2, 0x2, 0x297, 
    0x298, 0x7, 0x91, 0x2, 0x2, 0x298, 0x299, 0x5, 0x8e, 0x48, 0x2, 0x299, 
    0x29a, 0x7, 0x92, 0x2, 0x2, 0x29a, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x29b, 
    0x29c, 0x7, 0x70, 0x2, 0x2, 0x29c, 0x29d, 0x7, 0x91, 0x2, 0x2, 0x29d, 
    0x29e, 0x5, 0x8e, 0x48, 0x2, 0x29e, 0x29f, 0x7, 0x92, 0x2, 0x2, 0x29f, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2a0, 0x2a1, 0x7, 0x71, 0x2, 0x2, 0x2a1, 
    0x2a2, 0x7, 0x91, 0x2, 0x2, 0x2a2, 0x2a3, 0x5, 0x8e, 0x48, 0x2, 0x2a3, 
    0x2a4, 0x7, 0x92, 0x2, 0x2, 0x2a4, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2a5, 
    0x2a6, 0x7, 0x73, 0x2, 0x2, 0x2a6, 0x2a7, 0x7, 0x91, 0x2, 0x2, 0x2a7, 
    0x2a8, 0x5, 0x8e, 0x48, 0x2, 0x2a8, 0x2a9, 0x7, 0x92, 0x2, 0x2, 0x2a9, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2aa, 0x2ab, 0x7, 0x75, 0x2, 0x2, 0x2ab, 
    0x2ac, 0x7, 0x91, 0x2, 0x2, 0x2ac, 0x2ad, 0x5, 0x8e, 0x48, 0x2, 0x2ad, 
    0x2ae, 0x7, 0x92, 0x2, 0x2, 0x2ae, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2af, 
    0x2b0, 0x7, 0x74, 0x2, 0x2, 0x2b0, 0x2b1, 0x7, 0x91, 0x2, 0x2, 0x2b1, 
    0x2b2, 0x5, 0x8e, 0x48, 0x2, 0x2b2, 0x2b3, 0x7, 0x92, 0x2, 0x2, 0x2b3, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2b4, 0x2b5, 0x7, 0x1d, 0x2, 0x2, 0x2b5, 
    0x2b6, 0x7, 0x91, 0x2, 0x2, 0x2b6, 0x2b7, 0x5, 0x8e, 0x48, 0x2, 0x2b7, 
    0x2b8, 0x7, 0x92, 0x2, 0x2, 0x2b8, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2b9, 
    0x2ba, 0x7, 0x58, 0x2, 0x2, 0x2ba, 0x2bb, 0x7, 0x91, 0x2, 0x2, 0x2bb, 
    0x2bc, 0x5, 0x8e, 0x48, 0x2, 0x2bc, 0x2bd, 0x7, 0x92, 0x2, 0x2, 0x2bd, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2be, 0x2bf, 0x7, 0x59, 0x2, 0x2, 0x2bf, 
    0x2c0, 0x7, 0x91, 0x2, 0x2, 0x2c0, 0x2c1, 0x5, 0x8e, 0x48, 0x2, 0x2c1, 
    0x2c2, 0x7, 0x92, 0x2, 0x2, 0x2c2, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2c3, 
    0x2c4, 0x7, 0x5a, 0x2, 0x2, 0x2c4, 0x2c5, 0x7, 0x91, 0x2, 0x2, 0x2c5, 
    0x2c6, 0x5, 0x8e, 0x48, 0x2, 0x2c6, 0x2c7, 0x7, 0x92, 0x2, 0x2, 0x2c7, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2c8, 0x2c9, 0x7, 0x5b, 0x2, 0x2, 0x2c9, 
    0x2ca, 0x7, 0x91, 0x2, 0x2, 0x2ca, 0x2cb, 0x5, 0x8e, 0x48, 0x2, 0x2cb, 
    0x2cc, 0x7, 0x92, 0x2, 0x2, 0x2cc, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2cd, 
    0x2ce, 0x7, 0x5c, 0x2, 0x2, 0x2ce, 0x2cf, 0x7, 0x91, 0x2, 0x2, 0x2cf, 
    0x2d0, 0x5, 0x8e, 0x48, 0x2, 0x2d0, 0x2d1, 0x7, 0x92, 0x2, 0x2, 0x2d1, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2d2, 0x2d3, 0x7, 0x5d, 0x2, 0x2, 0x2d3, 
    0x2d4, 0x7, 0x91, 0x2, 0x2, 0x2d4, 0x2d5, 0x5, 0x8e, 0x48, 0x2, 0x2d5, 
    0x2d6, 0x7, 0x92, 0x2, 0x2, 0x2d6, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2d7, 
    0x2d8, 0x7, 0x5e, 0x2, 0x2, 0x2d8, 0x2d9, 0x7, 0x91, 0x2, 0x2, 0x2d9, 
    0x2da, 0x5, 0x8e, 0x48, 0x2, 0x2da, 0x2db, 0x7, 0x92, 0x2, 0x2, 0x2db, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2dc, 0x2dd, 0x7, 0x5f, 0x2, 0x2, 0x2dd, 
    0x2de, 0x7, 0x91, 0x2, 0x2, 0x2de, 0x2df, 0x5, 0x8e, 0x48, 0x2, 0x2df, 
    0x2e0, 0x7, 0x92, 0x2, 0x2, 0x2e0, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2e1, 
    0x2e2, 0x7, 0x76, 0x2, 0x2, 0x2e2, 0x2e3, 0x7, 0x91, 0x2, 0x2, 0x2e3, 
    0x2e4, 0x5, 0x8e, 0x48, 0x2, 0x2e4, 0x2e5, 0x7, 0x92, 0x2, 0x2, 0x2e5, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2e6, 0x2e7, 0x7, 0x77, 0x2, 0x2, 0x2e7, 
    0x2e8, 0x7, 0x91, 0x2, 0x2, 0x2e8, 0x2e9, 0x5, 0x8e, 0x48, 0x2, 0x2e9, 
    0x2ea, 0x7, 0x92, 0x2, 0x2, 0x2ea, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2eb, 
    0x2ec, 0x7, 0x78, 0x2, 0x2, 0x2ec, 0x2ed, 0x7, 0x91, 0x2, 0x2, 0x2ed, 
    0x2ee, 0x5, 0x8e, 0x48, 0x2, 0x2ee, 0x2ef, 0x7, 0x92, 0x2, 0x2, 0x2ef, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2f0, 0x2f1, 0x7, 0x79, 0x2, 0x2, 0x2f1, 
    0x2f2, 0x7, 0x91, 0x2, 0x2, 0x2f2, 0x2f3, 0x5, 0x8e, 0x48, 0x2, 0x2f3, 
    0x2f4, 0x7, 0x92, 0x2, 0x2, 0x2f4, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2f5, 
    0x2f6, 0x7, 0x7a, 0x2, 0x2, 0x2f6, 0x2f7, 0x7, 0x91, 0x2, 0x2, 0x2f7, 
    0x2f8, 0x5, 0x8e, 0x48, 0x2, 0x2f8, 0x2f9, 0x7, 0x92, 0x2, 0x2, 0x2f9, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2fa, 0x2fb, 0x7, 0x7b, 0x2, 0x2, 0x2fb, 
    0x2fc, 0x7, 0x91, 0x2, 0x2, 0x2fc, 0x2fd, 0x5, 0x8e, 0x48, 0x2, 0x2fd, 
    0x2fe, 0x7, 0x92, 0x2, 0x2, 0x2fe, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x2ff, 
    0x300, 0x7, 0x6a, 0x2, 0x2, 0x300, 0x301, 0x7, 0x91, 0x2, 0x2, 0x301, 
    0x302, 0x5, 0x8e, 0x48, 0x2, 0x302, 0x303, 0x7, 0xd, 0x2, 0x2, 0x303, 
    0x304, 0x5, 0x8e, 0x48, 0x2, 0x304, 0x305, 0x7, 0x92, 0x2, 0x2, 0x305, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x306, 0x307, 0x7, 0x6c, 0x2, 0x2, 0x307, 
    0x308, 0x7, 0x91, 0x2, 0x2, 0x308, 0x309, 0x5, 0x8e, 0x48, 0x2, 0x309, 
    0x30a, 0x7, 0xd, 0x2, 0x2, 0x30a, 0x30b, 0x5, 0x8e, 0x48, 0x2, 0x30b, 
    0x30c, 0x7, 0x92, 0x2, 0x2, 0x30c, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x30d, 
    0x30e, 0x7, 0x6e, 0x2, 0x2, 0x30e, 0x30f, 0x7, 0x91, 0x2, 0x2, 0x30f, 
    0x310, 0x5, 0x8e, 0x48, 0x2, 0x310, 0x311, 0x7, 0xd, 0x2, 0x2, 0x311, 
    0x312, 0x5, 0x8e, 0x48, 0x2, 0x312, 0x313, 0x7, 0x92, 0x2, 0x2, 0x313, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x314, 0x315, 0x7, 0x72, 0x2, 0x2, 0x315, 
    0x316, 0x7, 0x91, 0x2, 0x2, 0x316, 0x317, 0x5, 0x8e, 0x48, 0x2, 0x317, 
    0x318, 0x7, 0xd, 0x2, 0x2, 0x318, 0x319, 0x5, 0x8e, 0x48, 0x2, 0x319, 
    0x31a, 0x7, 0x92, 0x2, 0x2, 0x31a, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x31b, 
    0x31c, 0x7, 0x73, 0x2, 0x2, 0x31c, 0x31d, 0x7, 0x91, 0x2, 0x2, 0x31d, 
    0x31e, 0x5, 0x8e, 0x48, 0x2, 0x31e, 0x31f, 0x7, 0xd, 0x2, 0x2, 0x31f, 
    0x320, 0x5, 0x8e, 0x48, 0x2, 0x320, 0x321, 0x7, 0x92, 0x2, 0x2, 0x321, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x322, 0x323, 0x7, 0x15, 0x2, 0x2, 0x323, 
    0x324, 0x7, 0x91, 0x2, 0x2, 0x324, 0x325, 0x5, 0x8e, 0x48, 0x2, 0x325, 
    0x326, 0x7, 0xd, 0x2, 0x2, 0x326, 0x327, 0x5, 0x8e, 0x48, 0x2, 0x327, 
    0x328, 0x7, 0x92, 0x2, 0x2, 0x328, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x329, 
    0x32a, 0x7, 0x81, 0x2, 0x2, 0x32a, 0x32b, 0x7, 0x91, 0x2, 0x2, 0x32b, 
    0x32c, 0x5, 0x8e, 0x48, 0x2, 0x32c, 0x32d, 0x7, 0xd, 0x2, 0x2, 0x32d, 
    0x32e, 0x5, 0x8e, 0x48, 0x2, 0x32e, 0x32f, 0x7, 0x92, 0x2, 0x2, 0x32f, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x330, 0x331, 0x7, 0x82, 0x2, 0x2, 0x331, 
    0x332, 0x7, 0x91, 0x2, 0x2, 0x332, 0x333, 0x5, 0x8e, 0x48, 0x2, 0x333, 
    0x334, 0x7, 0xd, 0x2, 0x2, 0x334, 0x335, 0x5, 0x8e, 0x48, 0x2, 0x335, 
    0x336, 0x7, 0x92, 0x2, 0x2, 0x336, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x337, 
    0x338, 0x7, 0x83, 0x2, 0x2, 0x338, 0x339, 0x7, 0x91, 0x2, 0x2, 0x339, 
    0x33a, 0x5, 0x8e, 0x48, 0x2, 0x33a, 0x33b, 0x7, 0xd, 0x2, 0x2, 0x33b, 
    0x33c, 0x5, 0x8e, 0x48, 0x2, 0x33c, 0x33d, 0x7, 0x92, 0x2, 0x2, 0x33d, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x33e, 0x33f, 0x7, 0x84, 0x2, 0x2, 0x33f, 
    0x340, 0x7, 0x91, 0x2, 0x2, 0x340, 0x341, 0x5, 0x8e, 0x48, 0x2, 0x341, 
    0x342, 0x7, 0xd, 0x2, 0x2, 0x342, 0x343, 0x5, 0x8e, 0x48, 0x2, 0x343, 
    0x344, 0x7, 0x92, 0x2, 0x2, 0x344, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x345, 
    0x346, 0x7, 0x85, 0x2, 0x2, 0x346, 0x347, 0x7, 0x91, 0x2, 0x2, 0x347, 
    0x348, 0x5, 0x8e, 0x48, 0x2, 0x348, 0x349, 0x7, 0xd, 0x2, 0x2, 0x349, 
    0x34a, 0x5, 0x8e, 0x48, 0x2, 0x34a, 0x34b, 0x7, 0x92, 0x2, 0x2, 0x34b, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x34c, 0x34d, 0x7, 0x86, 0x2, 0x2, 0x34d, 
    0x34e, 0x7, 0x91, 0x2, 0x2, 0x34e, 0x34f, 0x5, 0x8e, 0x48, 0x2, 0x34f, 
    0x350, 0x7, 0xd, 0x2, 0x2, 0x350, 0x351, 0x5, 0x8e, 0x48, 0x2, 0x351, 
    0x352, 0x7, 0x92, 0x2, 0x2, 0x352, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x353, 
    0x354, 0x7, 0x87, 0x2, 0x2, 0x354, 0x355, 0x7, 0x91, 0x2, 0x2, 0x355, 
    0x356, 0x5, 0x8e, 0x48, 0x2, 0x356, 0x357, 0x7, 0xd, 0x2, 0x2, 0x357, 
    0x358, 0x5, 0x8e, 0x48, 0x2, 0x358, 0x359, 0x7, 0x92, 0x2, 0x2, 0x359, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x35a, 0x35b, 0x7, 0x7e, 0x2, 0x2, 0x35b, 
    0x35c, 0x7, 0x91, 0x2, 0x2, 0x35c, 0x35d, 0x5, 0x8e, 0x48, 0x2, 0x35d, 
    0x35e, 0x7, 0xd, 0x2, 0x2, 0x35e, 0x35f, 0x5, 0x8e, 0x48, 0x2, 0x35f, 
    0x360, 0x7, 0x92, 0x2, 0x2, 0x360, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x361, 
    0x362, 0x7, 0x7c, 0x2, 0x2, 0x362, 0x363, 0x7, 0x91, 0x2, 0x2, 0x363, 
    0x364, 0x5, 0x8e, 0x48, 0x2, 0x364, 0x365, 0x7, 0xd, 0x2, 0x2, 0x365, 
    0x366, 0x5, 0x8e, 0x48, 0x2, 0x366, 0x367, 0x7, 0x92, 0x2, 0x2, 0x367, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x368, 0x369, 0x7, 0x7d, 0x2, 0x2, 0x369, 
    0x36a, 0x7, 0x91, 0x2, 0x2, 0x36a, 0x36b, 0x5, 0x8e, 0x48, 0x2, 0x36b, 
    0x36c, 0x7, 0xd, 0x2, 0x2, 0x36c, 0x36d, 0x5, 0x8e, 0x48, 0x2, 0x36d, 
    0x36e, 0x7, 0x92, 0x2, 0x2, 0x36e, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x36f, 
    0x370, 0x7, 0x7f, 0x2, 0x2, 0x370, 0x371, 0x7, 0x91, 0x2, 0x2, 0x371, 
    0x372, 0x5, 0x8e, 0x48, 0x2, 0x372, 0x373, 0x7, 0x45, 0x2, 0x2, 0x373, 
    0x374, 0x5, 0x90, 0x49, 0x2, 0x374, 0x375, 0x7, 0x92, 0x2, 0x2, 0x375, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x376, 0x377, 0x7, 0x91, 0x2, 0x2, 0x377, 
    0x378, 0x5, 0x8e, 0x48, 0x2, 0x378, 0x379, 0x7, 0x92, 0x2, 0x2, 0x379, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x37a, 0x3a1, 0x5, 0x74, 0x3b, 0x2, 0x37b, 
    0x3a1, 0x5, 0x92, 0x4a, 0x2, 0x37c, 0x3a1, 0x7, 0x3, 0x2, 0x2, 0x37d, 
    0x3a1, 0x7, 0xa1, 0x2, 0x2, 0x37e, 0x3a1, 0x7, 0x61, 0x2, 0x2, 0x37f, 
    0x3a1, 0x7, 0x60, 0x2, 0x2, 0x380, 0x3a1, 0x7, 0xa2, 0x2, 0x2, 0x381, 
    0x3a1, 0x7, 0x11, 0x2, 0x2, 0x382, 0x3a1, 0x7, 0x9e, 0x2, 0x2, 0x383, 
    0x384, 0x7, 0x55, 0x2, 0x2, 0x384, 0x385, 0x7, 0x91, 0x2, 0x2, 0x385, 
    0x386, 0x5, 0x8e, 0x48, 0x2, 0x386, 0x387, 0x7, 0x92, 0x2, 0x2, 0x387, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x388, 0x389, 0x7, 0x56, 0x2, 0x2, 0x389, 
    0x38a, 0x7, 0x91, 0x2, 0x2, 0x38a, 0x38b, 0x5, 0x8e, 0x48, 0x2, 0x38b, 
    0x38c, 0x7, 0x92, 0x2, 0x2, 0x38c, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x38d, 
    0x38e, 0x7, 0x54, 0x2, 0x2, 0x38e, 0x38f, 0x7, 0x91, 0x2, 0x2, 0x38f, 
    0x390, 0x5, 0x8e, 0x48, 0x2, 0x390, 0x391, 0x7, 0x92, 0x2, 0x2, 0x391, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x392, 0x393, 0x7, 0x57, 0x2, 0x2, 0x393, 
    0x394, 0x7, 0x91, 0x2, 0x2, 0x394, 0x395, 0x5, 0x8e, 0x48, 0x2, 0x395, 
    0x396, 0x7, 0x92, 0x2, 0x2, 0x396, 0x3a1, 0x3, 0x2, 0x2, 0x2, 0x397, 
    0x398, 0x7, 0x57, 0x2, 0x2, 0x398, 0x399, 0x7, 0x91, 0x2, 0x2, 0x399, 
    0x39a, 0x7, 0x8a, 0x2, 0x2, 0x39a, 0x3a1, 0x7, 0x92, 0x2, 0x2, 0x39b, 
    0x39c, 0x7, 0x53, 0x2, 0x2, 0x39c, 0x39d, 0x7, 0x91, 0x2, 0x2, 0x39d, 
    0x39e, 0x5, 0x8e, 0x48, 0x2, 0x39e, 0x39f, 0x7, 0x92, 0x2, 0x2, 0x39f, 
    0x3a1, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x25a, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x25d, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x25f, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x264, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x269, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x26e, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x273, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x278, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x27d, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x282, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x287, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x28c, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x291, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x296, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x29b, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2a0, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2a5, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2aa, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2af, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2b4, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2b9, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2be, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2c3, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2c8, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2cd, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2d2, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2d7, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2dc, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2e1, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2e6, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2eb, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2f0, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2f5, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x2fa, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x2ff, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x306, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x30d, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x314, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x31b, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x322, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x329, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x330, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x337, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x33e, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x345, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x34c, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x353, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x35a, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x361, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x368, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x36f, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x376, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x37a, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x37b, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x37c, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x37d, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x37e, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x37f, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x380, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x381, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x382, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x383, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x388, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x38d, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x392, 0x3, 0x2, 0x2, 0x2, 0x3a0, 
    0x397, 0x3, 0x2, 0x2, 0x2, 0x3a0, 0x39b, 0x3, 0x2, 0x2, 0x2, 0x3a1, 
    0x3d5, 0x3, 0x2, 0x2, 0x2, 0x3a2, 0x3a3, 0xc, 0x30, 0x2, 0x2, 0x3a3, 
    0x3a4, 0x9, 0x5, 0x2, 0x2, 0x3a4, 0x3d4, 0x5, 0x8e, 0x48, 0x31, 0x3a5, 
    0x3a6, 0xc, 0x2f, 0x2, 0x2, 0x3a6, 0x3a7, 0x9, 0x6, 0x2, 0x2, 0x3a7, 
    0x3d4, 0x5, 0x8e, 0x48, 0x30, 0x3a8, 0x3a9, 0xc, 0x2e, 0x2, 0x2, 0x3a9, 
    0x3aa, 0x7, 0x8c, 0x2, 0x2, 0x3aa, 0x3d4, 0x5, 0x8e, 0x48, 0x2f, 0x3ab, 
    0x3ac, 0xc, 0x28, 0x2, 0x2, 0x3ac, 0x3ad, 0x7, 0x8d, 0x2, 0x2, 0x3ad, 
    0x3d4, 0x5, 0x8e, 0x48, 0x29, 0x3ae, 0x3af, 0xc, 0x27, 0x2, 0x2, 0x3af, 
    0x3b0, 0x9, 0x7, 0x2, 0x2, 0x3b0, 0x3d4, 0x5, 0x8e, 0x48, 0x28, 0x3b1, 
    0x3b2, 0xc, 0x26, 0x2, 0x2, 0x3b2, 0x3b3, 0x9, 0x8, 0x2, 0x2, 0x3b3, 
    0x3d4, 0x5, 0x8e, 0x48, 0x27, 0x3b4, 0x3b5, 0xc, 0x25, 0x2, 0x2, 0x3b5, 
    0x3b6, 0x9, 0x9, 0x2, 0x2, 0x3b6, 0x3d4, 0x5, 0x8e, 0x48, 0x26, 0x3b7, 
    0x3b8, 0xc, 0x24, 0x2, 0x2, 0x3b8, 0x3b9, 0x9, 0xa, 0x2, 0x2, 0x3b9, 
    0x3d4, 0x5, 0x8e, 0x48, 0x25, 0x3ba, 0x3bb, 0xc, 0x23, 0x2, 0x2, 0x3bb, 
    0x3bc, 0x9, 0xb, 0x2, 0x2, 0x3bc, 0x3d4, 0x5, 0x8e, 0x48, 0x24, 0x3bd, 
    0x3be, 0xc, 0x22, 0x2, 0x2, 0x3be, 0x3bf, 0x7, 0x90, 0x2, 0x2, 0x3bf, 
    0x3d4, 0x5, 0x8e, 0x48, 0x23, 0x3c0, 0x3c1, 0xc, 0x16, 0x2, 0x2, 0x3c1, 
    0x3c2, 0x7, 0x4c, 0x2, 0x2, 0x3c2, 0x3c3, 0x5, 0x8e, 0x48, 0x2, 0x3c3, 
    0x3c4, 0x7, 0x99, 0x2, 0x2, 0x3c4, 0x3c5, 0x5, 0x8e, 0x48, 0x17, 0x3c5, 
    0x3d4, 0x3, 0x2, 0x2, 0x2, 0x3c6, 0x3c7, 0xc, 0x15, 0x2, 0x2, 0x3c7, 
    0x3c8, 0x7, 0x99, 0x2, 0x2, 0x3c8, 0x3d4, 0x5, 0x8e, 0x48, 0x16, 0x3c9, 
    0x3ca, 0xc, 0x14, 0x2, 0x2, 0x3ca, 0x3cb, 0x7, 0x98, 0x2, 0x2, 0x3cb, 
    0x3d4, 0x5, 0x8e, 0x48, 0x15, 0x3cc, 0x3cd, 0xc, 0x52, 0x2, 0x2, 0x3cd, 
    0x3ce, 0x7, 0x48, 0x2, 0x2, 0x3ce, 0x3d4, 0x7, 0x4a, 0x2, 0x2, 0x3cf, 
    0x3d0, 0xc, 0x51, 0x2, 0x2, 0x3d0, 0x3d1, 0x7, 0x48, 0x2, 0x2, 0x3d1, 
    0x3d2, 0x7, 0x49, 0x2, 0x2, 0x3d2, 0x3d4, 0x7, 0x4a, 0x2, 0x2, 0x3d3, 
    0x3a2, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3a5, 0x3, 0x2, 0x2, 0x2, 0x3d3, 
    0x3a8, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3ab, 0x3, 0x2, 0x2, 0x2, 0x3d3, 
    0x3ae, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3b1, 0x3, 0x2, 0x2, 0x2, 0x3d3, 
    0x3b4, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3b7, 0x3, 0x2, 0x2, 0x2, 0x3d3, 
    0x3ba, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3bd, 0x3, 0x2, 0x2, 0x2, 0x3d3, 
    0x3c0, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3c6, 0x3, 0x2, 0x2, 0x2, 0x3d3, 
    0x3c9, 0x3, 0x2, 0x2, 0x2, 0x3d3, 0x3cc, 0x3, 0x2, 0x2, 0x2, 0x3d3, 
    0x3cf, 0x3, 0x2, 0x2, 0x2, 0x3d4, 0x3d7, 0x3, 0x2, 0x2, 0x2, 0x3d5, 
    0x3d3, 0x3, 0x2, 0x2, 0x2, 0x3d5, 0x3d6, 0x3, 0x2, 0x2, 0x2, 0x3d6, 
    0x8f, 0x3, 0x2, 0x2, 0x2, 0x3d7, 0x3d5, 0x3, 0x2, 0x2, 0x2, 0x3d8, 0x3d9, 
    0x9, 0xc, 0x2, 0x2, 0x3d9, 0x91, 0x3, 0x2, 0x2, 0x2, 0x3da, 0x3e1, 0x5, 
    0x94, 0x4b, 0x2, 0x3db, 0x3e1, 0x5, 0x98, 0x4d, 0x2, 0x3dc, 0x3e1, 0x5, 
    0x96, 0x4c, 0x2, 0x3dd, 0x3e1, 0x5, 0x9a, 0x4e, 0x2, 0x3de, 0x3e1, 0x5, 
    0x9c, 0x4f, 0x2, 0x3df, 0x3e1, 0x5, 0x9e, 0x50, 0x2, 0x3e0, 0x3da, 0x3, 
    0x2, 0x2, 0x2, 0x3e0, 0x3db, 0x3, 0x2, 0x2, 0x2, 0x3e0, 0x3dc, 0x3, 
    0x2, 0x2, 0x2, 0x3e0, 0x3dd, 0x3, 0x2, 0x2, 0x2, 0x3e0, 0x3de, 0x3, 
    0x2, 0x2, 0x2, 0x3e0, 0x3df, 0x3, 0x2, 0x2, 0x2, 0x3e1, 0x93, 0x3, 0x2, 
    0x2, 0x2, 0x3e2, 0x3e3, 0x7, 0x15, 0x2, 0x2, 0x3e3, 0x3e4, 0x7, 0x91, 
    0x2, 0x2, 0x3e4, 0x3e5, 0x5, 0xa6, 0x54, 0x2, 0x3e5, 0x3e6, 0x7, 0x92, 
    0x2, 0x2, 0x3e6, 0x95, 0x3, 0x2, 0x2, 0x2, 0x3e7, 0x3e8, 0x7, 0x17, 
    0x2, 0x2, 0x3e8, 0x3e9, 0x5, 0xa4, 0x53, 0x2, 0x3e9, 0x97, 0x3, 0x2, 
    0x2, 0x2, 0x3ea, 0x3eb, 0x7, 0x19, 0x2, 0x2, 0x3eb, 0x3ec, 0x5, 0xa2, 
    0x52, 0x2, 0x3ec, 0x99, 0x3, 0x2, 0x2, 0x2, 0x3ed, 0x3ee, 0x7, 0x16, 
    0x2, 0x2, 0x3ee, 0x3ef, 0x7, 0x91, 0x2, 0x2, 0x3ef, 0x3f4, 0x5, 0xa0, 
    0x51, 0x2, 0x3f0, 0x3f1, 0x7, 0xd, 0x2, 0x2, 0x3f1, 0x3f3, 0x5, 0xa0, 
    0x51, 0x2, 0x3f2, 0x3f0, 0x3, 0x2, 0x2, 0x2, 0x3f3, 0x3f6, 0x3, 0x2, 
    0x2, 0x2, 0x3f4, 0x3f2, 0x3, 0x2, 0x2, 0x2, 0x3f4, 0x3f5, 0x3, 0x2, 
    0x2, 0x2, 0x3f5, 0x3f7, 0x3, 0x2, 0x2, 0x2, 0x3f6, 0x3f4, 0x3, 0x2, 
    0x2, 0x2, 0x3f7, 0x3f8, 0x7, 0x92, 0x2, 0x2, 0x3f8, 0x9b, 0x3, 0x2, 
    0x2, 0x2, 0x3f9, 0x3fa, 0x7, 0x18, 0x2, 0x2, 0x3fa, 0x3fb, 0x7, 0x91, 
    0x2, 0x2, 0x3fb, 0x400, 0x5, 0xa4, 0x53, 0x2, 0x3fc, 0x3fd, 0x7, 0xd, 
    0x2, 0x2, 0x3fd, 0x3ff, 0x5, 0xa4, 0x53, 0x2, 0x3fe, 0x3fc, 0x3, 0x2, 
    0x2, 0x2, 0x3ff, 0x402, 0x3, 0x2, 0x2, 0x2, 0x400, 0x3fe, 0x3, 0x2, 
    0x2, 0x2, 0x400, 0x401, 0x3, 0x2, 0x2, 0x2, 0x401, 0x403, 0x3, 0x2, 
    0x2, 0x2, 0x402, 0x400, 0x3, 0x2, 0x2, 0x2, 0x403, 0x404, 0x7, 0x92, 
    0x2, 0x2, 0x404, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x405, 0x406, 0x7, 0x1a, 
    0x2, 0x2, 0x406, 0x407, 0x7, 0x91, 0x2, 0x2, 0x407, 0x40c, 0x5, 0xa2, 
    0x52, 0x2, 0x408, 0x409, 0x7, 0xd, 0x2, 0x2, 0x409, 0x40b, 0x5, 0xa2, 
    0x52, 0x2, 0x40a, 0x408, 0x3, 0x2, 0x2, 0x2, 0x40b, 0x40e, 0x3, 0x2, 
    0x2, 0x2, 0x40c, 0x40a, 0x3, 0x2, 0x2, 0x2, 0x40c, 0x40d, 0x3, 0x2, 
    0x2, 0x2, 0x40d, 0x40f, 0x3, 0x2, 0x2, 0x2, 0x40e, 0x40c, 0x3, 0x2, 
    0x2, 0x2, 0x40f, 0x410, 0x7, 0x92, 0x2, 0x2, 0x410, 0x9f, 0x3, 0x2, 
    0x2, 0x2, 0x411, 0x417, 0x5, 0xa6, 0x54, 0x2, 0x412, 0x413, 0x7, 0x91, 
    0x2, 0x2, 0x413, 0x414, 0x5, 0xa6, 0x54, 0x2, 0x414, 0x415, 0x7, 0x92, 
    0x2, 0x2, 0x415, 0x417, 0x3, 0x2, 0x2, 0x2, 0x416, 0x411, 0x3, 0x2, 
    0x2, 0x2, 0x416, 0x412, 0x3, 0x2, 0x2, 0x2, 0x417, 0xa1, 0x3, 0x2, 0x2, 
    0x2, 0x418, 0x419, 0x7, 0x91, 0x2, 0x2, 0x419, 0x41e, 0x5, 0xa4, 0x53, 
    0x2, 0x41a, 0x41b, 0x7, 0xd, 0x2, 0x2, 0x41b, 0x41d, 0x5, 0xa4, 0x53, 
    0x2, 0x41c, 0x41a, 0x3, 0x2, 0x2, 0x2, 0x41d, 0x420, 0x3, 0x2, 0x2, 
    0x2, 0x41e, 0x41c, 0x3, 0x2, 0x2, 0x2, 0x41e, 0x41f, 0x3, 0x2, 0x2, 
    0x2, 0x41f, 0x421, 0x3, 0x2, 0x2, 0x2, 0x420, 0x41e, 0x3, 0x2, 0x2, 
    0x2, 0x421, 0x422, 0x7, 0x92, 0x2, 0x2, 0x422, 0xa3, 0x3, 0x2, 0x2, 
    0x2, 0x423, 0x424, 0x7, 0x91, 0x2, 0x2, 0x424, 0x429, 0x5, 0xa6, 0x54, 
    0x2, 0x425, 0x426, 0x7, 0xd, 0x2, 0x2, 0x426, 0x428, 0x5, 0xa6, 0x54, 
    0x2, 0x427, 0x425, 0x3, 0x2, 0x2, 0x2, 0x428, 0x42b, 0x3, 0x2, 0x2, 
    0x2, 0x429, 0x427, 0x3, 0x2, 0x2, 0x2, 0x429, 0x42a, 0x3, 0x2, 0x2, 
    0x2, 0x42a, 0x42c, 0x3, 0x2, 0x2, 0x2, 0x42b, 0x429, 0x3, 0x2, 0x2, 
    0x2, 0x42c, 0x42d, 0x7, 0x92, 0x2, 0x2, 0x42d, 0xa5, 0x3, 0x2, 0x2, 
    0x2, 0x42e, 0x430, 0x7, 0x89, 0x2, 0x2, 0x42f, 0x42e, 0x3, 0x2, 0x2, 
    0x2, 0x42f, 0x430, 0x3, 0x2, 0x2, 0x2, 0x430, 0x431, 0x3, 0x2, 0x2, 
    0x2, 0x431, 0x437, 0x7, 0xa1, 0x2, 0x2, 0x432, 0x434, 0x7, 0x89, 0x2, 
    0x2, 0x433, 0x432, 0x3, 0x2, 0x2, 0x2, 0x433, 0x434, 0x3, 0x2, 0x2, 
    0x2, 0x434, 0x435, 0x3, 0x2, 0x2, 0x2, 0x435, 0x437, 0x7, 0xa2, 0x2, 
    0x2, 0x436, 0x42f, 0x3, 0x2, 0x2, 0x2, 0x436, 0x433, 0x3, 0x2, 0x2, 
    0x2, 0x437, 0x440, 0x3, 0x2, 0x2, 0x2, 0x438, 0x43a, 0x7, 0x89, 0x2, 
    0x2, 0x439, 0x438, 0x3, 0x2, 0x2, 0x2, 0x439, 0x43a, 0x3, 0x2, 0x2, 
    0x2, 0x43a, 0x43b, 0x3, 0x2, 0x2, 0x2, 0x43b, 0x441, 0x7, 0xa1, 0x2, 
    0x2, 0x43c, 0x43e, 0x7, 0x89, 0x2, 0x2, 0x43d, 0x43c, 0x3, 0x2, 0x2, 
    0x2, 0x43d, 0x43e, 0x3, 0x2, 0x2, 0x2, 0x43e, 0x43f, 0x3, 0x2, 0x2, 
    0x2, 0x43f, 0x441, 0x7, 0xa2, 0x2, 0x2, 0x440, 0x439, 0x3, 0x2, 0x2, 
    0x2, 0x440, 0x43d, 0x3, 0x2, 0x2, 0x2, 0x441, 0xa7, 0x3, 0x2, 0x2, 0x2, 
    0x3e, 0xab, 0xbb, 0xc1, 0xcb, 0xd5, 0xdf, 0xee, 0xf2, 0xf7, 0xfc, 0x100, 
    0x104, 0x10c, 0x119, 0x14c, 0x151, 0x158, 0x15d, 0x168, 0x173, 0x1a3, 
    0x1ad, 0x1b2, 0x1b6, 0x1bc, 0x1c0, 0x1cb, 0x1d0, 0x1d7, 0x1df, 0x1e7, 
    0x1ef, 0x1f7, 0x201, 0x207, 0x20a, 0x216, 0x223, 0x228, 0x22f, 0x244, 
    0x248, 0x250, 0x256, 0x3a0, 0x3d3, 0x3d5, 0x3e0, 0x3f4, 0x400, 0x40c, 
    0x416, 0x41e, 0x429, 0x42f, 0x433, 0x436, 0x439, 0x43d, 0x440, 
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

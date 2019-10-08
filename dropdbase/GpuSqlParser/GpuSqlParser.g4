parser grammar GpuSqlParser;

options {
	tokenVocab = GpuSqlLexer;
}

sqlFile: statement* EOF;

statement:
	sqlSelect
	| sqlCreateDb
	| sqlDropDb
	| sqlCreateTable
	| sqlDropTable
	| sqlAlterTable
	| sqlAlterDatabase
	| sqlCreateIndex
	| sqlInsertInto
	| showStatement;

showStatement: (showDatabases | showTables | showColumns);

showDatabases: SHOWDB SEMICOL;
showTables: SHOWTB ((FROM | IN) database)? SEMICOL;
showColumns:
	SHOWCL (FROM | IN) table ((FROM | IN) database)? SEMICOL;

sqlSelect:
	SELECT selectColumns FROM fromTables (joinClauses)? (
		WHERE whereClause
	)? (GROUPBY groupByColumns)? (ORDERBY orderByColumns)? (
		LIMIT limit
	)? (OFFSET offset)? SEMICOL;
sqlCreateDb: CREATEDB database (blockSize)? SEMICOL;
sqlDropDb: DROPDB database SEMICOL;
sqlCreateTable:
	CREATETABLE table LPAREN newTableEntries RPAREN SEMICOL;
sqlDropTable: DROPTABLE table SEMICOL;
sqlAlterTable: ALTERTABLE table alterTableEntries SEMICOL;
sqlAlterDatabase: ALTERDATABASE database alterDatabaseEntries SEMICOL;
sqlCreateIndex:
	CREATEINDEX indexName ON table LPAREN indexColumns RPAREN SEMICOL;
sqlInsertInto:
	INSERTINTO table LPAREN insertIntoColumns RPAREN VALUES LPAREN insertIntoValues RPAREN SEMICOL;
newTableEntries: ((newTableEntry (COMMA newTableEntry)*));
newTableEntry: (newTableColumn | newTableIndex);
alterDatabaseEntries: ((alterDatabaseEntry (COMMA alterDatabaseEntry)*));
alterDatabaseEntry: (renameDatabase);
renameDatabase: (RENAMETO database);
alterTableEntries: ((alterTableEntry (COMMA alterTableEntry)*));
alterTableEntry: (addColumn | dropColumn | alterColumn | renameColumn | renameTable);
addColumn: (ADD column datatype);
dropColumn: (DROPCOLUMN column);
alterColumn: (ALTERCOLUMN column datatype);
renameColumn: (RENAMECOLUMN renameColumnFrom TO renameColumnTo);
renameTable: (RENAMETO table);
renameColumnFrom: column;
renameColumnTo: column;
newTableColumn: (column datatype);
newTableIndex: (INDEX indexName LPAREN indexColumns RPAREN);
selectColumns: (
		(
			(selectColumn | selectAllColumns) (
				COMMA (selectColumn | selectAllColumns)
			)*
		)
	);
selectColumn: expression (AS alias)?;
selectAllColumns: ASTERISK;
whereClause: expression;
orderByColumns: ((orderByColumn (COMMA orderByColumn)*));
orderByColumn: (expression DIR?);
insertIntoValues: ((columnValue (COMMA columnValue)*));
insertIntoColumns: ((columnId (COMMA columnId)*));
indexColumns: ((column (COMMA column)*));
groupByColumns: ((groupByColumn (COMMA groupByColumn)*));
groupByColumn: expression;
fromTables: ((fromTable (COMMA fromTable)*));
joinClauses: (joinClause)+;
joinClause: (
		(joinType)? JOIN joinTable ON joinColumnLeft joinOperator joinColumnRight
	);
joinTable: table (AS alias)?;
joinColumnLeft: columnId;
joinColumnRight: columnId;
joinOperator: (
		GREATER
		| LESS
		| GREATEREQ
		| LESSEQ
		| EQUALS
		| NOTEQUALS
		| NOTEQUALS_GT_LT
	);
joinType: (INNER | LEFT | RIGHT | FULLOUTER);
fromTable: table (AS alias)?;
columnId: (column) | (table DOT column);
table: (ID | DELIMID);
column: (ID | DELIMID);
database: (ID | DELIMID);
alias: (ID | DELIMID);
indexName: (ID | DELIMID);
limit: INTLIT;
offset: INTLIT;
blockSize: INTLIT;
columnValue: (
		MINUS? INTLIT
		| MINUS? FLOATLIT
		| geometry
		| NULLLIT
		| STRING
		| DATETIMELIT
		| BOOLEANLIT
	);

expression:
	op = LOGICAL_NOT expression														# unaryOperation
	| op = MINUS expression															# unaryOperation
	| expression op = ISNULL														# unaryOperation
	| expression op = ISNOTNULL														# unaryOperation
	| op = ABS LPAREN expression RPAREN												# unaryOperation
	| op = SIN LPAREN expression RPAREN												# unaryOperation
	| op = COS LPAREN expression RPAREN												# unaryOperation
	| op = TAN LPAREN expression RPAREN												# unaryOperation
	| op = COT LPAREN expression RPAREN												# unaryOperation
	| op = ASIN LPAREN expression RPAREN											# unaryOperation
	| op = ACOS LPAREN expression RPAREN											# unaryOperation
	| op = ATAN LPAREN expression RPAREN											# unaryOperation
	| op = LOG10 LPAREN expression RPAREN											# unaryOperation
	| op = LOG LPAREN expression RPAREN												# unaryOperation
	| op = EXP LPAREN expression RPAREN												# unaryOperation
	| op = SQRT LPAREN expression RPAREN											# unaryOperation
	| op = SQUARE LPAREN expression RPAREN											# unaryOperation
	| op = SIGN LPAREN expression RPAREN											# unaryOperation
	| op = ROUND LPAREN expression RPAREN											# unaryOperation
	| op = FLOOR LPAREN expression RPAREN											# unaryOperation
	| op = CEIL LPAREN expression RPAREN											# unaryOperation
	| op = DATETYPE LPAREN expression RPAREN									    # unaryOperation
	| op = YEAR LPAREN expression RPAREN											# unaryOperation
	| op = MONTH LPAREN expression RPAREN											# unaryOperation
	| op = DAY LPAREN expression RPAREN												# unaryOperation
	| op = HOUR LPAREN expression RPAREN											# unaryOperation
	| op = MINUTE LPAREN expression RPAREN											# unaryOperation
	| op = SECOND LPAREN expression RPAREN											# unaryOperation
	| op = LTRIM LPAREN expression RPAREN											# unaryOperation
	| op = RTRIM LPAREN expression RPAREN											# unaryOperation
	| op = LOWER LPAREN expression RPAREN											# unaryOperation
	| op = UPPER LPAREN expression RPAREN											# unaryOperation
	| op = REVERSE LPAREN expression RPAREN											# unaryOperation
	| op = LEN LPAREN expression RPAREN												# unaryOperation
	| left = expression op = (DIVISION | ASTERISK) right = expression				# binaryOperation
	| left = expression op = (PLUS | MINUS) right = expression						# binaryOperation
	| left = expression op = MODULO right = expression								# binaryOperation
	| op = ATAN2 LPAREN left = expression COMMA right = expression RPAREN			# binaryOperation
	| op = LOG LPAREN left = expression COMMA right = expression RPAREN				# binaryOperation
	| op = POW LPAREN left = expression COMMA right = expression RPAREN				# binaryOperation
	| op = ROOT LPAREN left = expression COMMA right = expression RPAREN			# binaryOperation
	| op = ROUND LPAREN left = expression COMMA right = expression RPAREN			# binaryOperation
	| left = expression op = XOR right = expression									# binaryOperation
	| left = expression op = (BIT_AND | BIT_OR) right = expression					# binaryOperation
	| left = expression op = (L_SHIFT | R_SHIFT) right = expression					# binaryOperation
	| left = expression op = (GREATER | LESS) right = expression					# binaryOperation
	| left = expression op = (GREATEREQ | LESSEQ) right = expression				# binaryOperation
	| left = expression op = (EQUALS | NOTEQUALS) right = expression				# binaryOperation
	| left = expression op = NOTEQUALS_GT_LT right = expression						# binaryOperation
	| op = POINT LPAREN left = expression COMMA right = expression RPAREN			# binaryOperation
	| op = GEO_CONTAINS LPAREN left = expression COMMA right = expression RPAREN	# binaryOperation
	| op = GEO_INTERSECT LPAREN left = expression COMMA right = expression RPAREN	# binaryOperation
	| op = GEO_UNION LPAREN left = expression COMMA right = expression RPAREN		# binaryOperation
	| op = CONCAT LPAREN left = expression COMMA right = expression RPAREN			# binaryOperation
	| op = LEFT LPAREN left = expression COMMA right = expression RPAREN			# binaryOperation
	| op = RIGHT LPAREN left = expression COMMA right = expression RPAREN			# binaryOperation
	| expression op = BETWEEN expression op2 = AND expression						# ternaryOperation
	| left = expression op = AND right = expression									# binaryOperation
	| left = expression op = OR right = expression									# binaryOperation
	| op = CAST LPAREN expression AS datatype RPAREN								# castOperation
	| LPAREN expression RPAREN														# parenExpression
	| columnId																		# varReference
	| geometry																		# geoReference
	| DATETIMELIT																	# dateTimeLiteral
	| FLOATLIT																		# decimalLiteral
	| PI																			# piLiteral
	| NOW																			# nowLiteral
	| INTLIT																		# intLiteral
	| STRING																		# stringLiteral
	| BOOLEANLIT																	# booleanLiteral
	| op = MIN_AGG LPAREN (expression) RPAREN										# aggregation
	| op = MAX_AGG LPAREN (expression) RPAREN										# aggregation
	| op = SUM_AGG LPAREN (expression) RPAREN										# aggregation
	| op = COUNT_AGG LPAREN (expression) RPAREN										# aggregation
	| op = COUNT_AGG LPAREN ASTERISK RPAREN											# aggregation
	| op = AVG_AGG LPAREN (expression) RPAREN										# aggregation;

datatype: (
        INTTYPE
		| LONGTYPE
		| DATETYPE
		| DETETIMETYPE
		| FLOATTYPE
		| DOUBLETYPE
		| STRINGTYPE
		| BOOLEANTYPE
		| POINTTYPE
		| POLYTYPE
);

geometry: (
		pointGeometry
		| polygonGeometry
		| lineStringGeometry
		| multiPointGeometry
		| multiLineStringGeometry
		| multiPolygonGeometry
	);
pointGeometry: POINT LPAREN point RPAREN;
lineStringGeometry: LINESTRING lineString;
polygonGeometry: POLYGON polygon;
multiPointGeometry:
	MULTIPOINT LPAREN pointOrClosedPoint (
		COMMA pointOrClosedPoint
	)* RPAREN;
multiLineStringGeometry:
	MULTILINESTRING LPAREN lineString (COMMA lineString)* RPAREN;
multiPolygonGeometry:
	MULTIPOLYGON LPAREN polygon (COMMA polygon)* RPAREN;
pointOrClosedPoint: point | LPAREN point RPAREN;

polygon: LPAREN lineString (COMMA lineString)* RPAREN;
lineString: LPAREN point (COMMA point)* RPAREN;
point: (MINUS? FLOATLIT | MINUS? INTLIT) (
		MINUS? FLOATLIT
		| MINUS? INTLIT
	);
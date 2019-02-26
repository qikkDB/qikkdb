parser grammar GpuSqlParser;

options { tokenVocab=GpuSqlLexer; }

sqlFile     : statement* EOF ;

statement   : sqlSelect|sqlCreateDb|sqlCreateTable|sqlInsertInto|showStatement;

showStatement   : (showDatabases|showTables|showColumns);

showDatabases   : SHOWDB SEMICOL;
showTables      : SHOWTB ((FROM|IN) database)? SEMICOL;
showColumns     : SHOWCL (FROM|IN) table ((FROM|IN) database)? SEMICOL;

sqlSelect       : SELECT selectColumns FROM fromTables (joinClauses)? (WHERE whereClause)? (GROUPBY groupByColumns)? (ORDERBY orderByColumns)? (LIMIT limit)? (OFFSET offset)? SEMICOL;
sqlCreateDb     : CREATEDB database SEMICOL;
sqlCreateTable  : CREATETABLE table LPAREN newTableColumns RPAREN;
sqlInsertInto   : INSERTINTO table LPAREN insertIntoColumns RPAREN VALUES LPAREN insertIntoValues RPAREN SEMICOL;

newTableColumns     : ((newTableColumn (COMMA newTableColumn)*));
newTableColumn      : (columnId DATATYPE);
selectColumns       : (((selectColumn) (COMMA selectColumn)*));
selectColumn        : expression;
whereClause         : expression;
orderByColumns      : ((orderByColumn (COMMA orderByColumn)*));
orderByColumn       : (columnId DIR?);
insertIntoValues    : ((columnValue (COMMA columnValue)*));
insertIntoColumns   : ((columnId (COMMA columnId)*));
groupByColumns      : ((groupByColumn (COMMA groupByColumn)*));
groupByColumn       : expression;
columnId            : (column)|(table DOT column);
fromTables          : ((table (COMMA table)*));
joinClauses         : (joinClause)+;
joinClause          : (JOIN joinTable ON expression);
joinTable           : table;
table               : ID;
column              : ID;
database            : ID;
limit               : INTLIT;
offset              : INTLIT;
columnValue         : (INTLIT|FLOATLIT|geometry|STRINGLIT|);

expression : op=NOT expression                                                            # unaryOperation
           | op=MINUS expression                                                          # unaryOperation
           | op=YEAR LPAREN expression RPAREN                                             # unaryOperation
           | op=MONTH LPAREN expression RPAREN                                            # unaryOperation
           | op=DAY LPAREN expression RPAREN                                              # unaryOperation
           | op=HOUR LPAREN expression RPAREN                                             # unaryOperation
           | op=MINUTE LPAREN expression RPAREN                                           # unaryOperation
           | op=SECOND LPAREN expression RPAREN                                           # unaryOperation
           | left=expression op=(DIVISION|ASTERISK) right=expression                      # binaryOperation
           | left=expression op=(PLUS|MINUS) right=expression                             # binaryOperation
           | left=expression op=(GREATER|LESS) right=expression                           # binaryOperation
           | left=expression op=(GREATEREQ|LESSEQ) right=expression                       # binaryOperation
           | left=expression op=(EQUALS|NOTEQUALS) right=expression                       # binaryOperation
           | left=expression op=MODULO right=expression                                   # binaryOperation
           | left=expression op=GEO right=expression                                      # binaryOperation
           | expression op=BETWEEN expression op2=AND expression                          # ternaryOperation
           | left=expression op=AND right=expression                                      # binaryOperation
           | left=expression op=OR right=expression                                       # binaryOperation
           | LPAREN expression RPAREN                                                     # parenExpression
           | columnId                                                                     # varReference
           | geometry                                                                     # geoReference
           | DATETIMELIT                                                                  # dateTimeLiteral
           | FLOATLIT                                                                     # decimalLiteral
           | INTLIT                                                                       # intLiteral
           | STRINGLIT                                                                    # stringLiteral
           | BOOLEANLIT                                                                   # booleanLiteral
           | AGG LPAREN expression RPAREN                                                 # aggregation;

geometry : (pointGeometry | polygonGeometry | lineStringGeometry | multiPointGeometry | multiLineStringGeometry | multiPolygonGeometry);
pointGeometry           : POINT LPAREN point RPAREN;
lineStringGeometry      : LINESTRING lineString;
polygonGeometry         : POLYGON polygon;
multiPointGeometry      : MULTIPOINT LPAREN pointOrClosedPoint (COMMA pointOrClosedPoint)* RPAREN;
multiLineStringGeometry : MULTILINESTRING LPAREN lineString (COMMA lineString)* RPAREN;
multiPolygonGeometry    : MULTIPOLYGON LPAREN polygon (COMMA polygon)* RPAREN;
pointOrClosedPoint      : point | LPAREN point RPAREN;

polygon         : LPAREN lineString (COMMA lineString)* RPAREN;
lineString      : LPAREN point (COMMA point)* RPAREN;
point           : (FLOATLIT|INTLIT)(FLOATLIT|INTLIT);
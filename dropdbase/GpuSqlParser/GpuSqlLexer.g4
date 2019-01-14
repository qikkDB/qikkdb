lexer grammar GpuSqlLexer;

LF          : '\n' -> skip;
CR          : '\r' -> skip;
CRLF        : '\r\n' -> skip;
WS          : [\t\r\n ]+ -> skip ;
SEMICOL     : ';';
COMMA       : ',';
DOT         : '.';

DATATYPE    : (INTTYPE|FLOATTYPE|STRINGTYPE|BOOLEANTYPE);

POINT           : 'POINT';
MULTIPOINT      : 'MULTIPOINT';
LINESTRING      : 'LINESTRING';
MULTILINESTRING : 'MULTILINESTRING';
POLYGON         : 'POLYGON';
MULTIPOLYGON    : 'MULTIPOLYGON';

INTTYPE     : I N T;
LONGTYPE    : L O N G;
FLOATTYPE   : F L O A T;
DOUBLETYPE  : D O U B L E;
STRINGTYPE  : S T R I N G;
BOOLEANTYPE : B O O L E A N;
POINTTYPE   : P O I N T;
POLYTYPE    : P O L Y G O N;

INSERTINTO  : I N S E R T ' ' I N T O;
CREATEDB    : C R E A T E ' ' D A T A B A S E;
CREATETABLE : C R E A T E ' ' T A B L E;
VALUES      : V A L U E S;
SELECT      : S E L E C T;
FROM        : F R O M;
JOIN        : J O I N;
WHERE       : W H E R E;
GROUPBY     : G R O U P ' ' B Y;
AS          : A S;
IN          : I N;
BETWEEN     : B E T W E E N;
ON          : O N;
ORDERBY     : O R D E R ' ' B Y;
DIR         : (A S C) | (D E S C);
LIMIT       : L I M I T;
OFFSET      : O F F S E T;

SHOWDB      : S H O W ' ' D A T A B A S E S;
SHOWTB      : S H O W ' ' T A B L E S;
SHOWCL      : S H O W ' ' C O L U M N S;

AGG         : (MIN|MAX|AVG|SUM|COUNT);

AVG         : A V G;
SUM         : S U M;
MIN         : M I N;
MAX         : M A X;
COUNT       : C O U N T;

GEO             : (CONTAINS);

CONTAINS        : C O N T A I N S;

PLUS        : '+' ;
MINUS       : '-' ;
ASTERISK    : '*' ;
DIVISION    : '/' ;
MODULO      : '%' ;
EQUALS      : '=' ;
NOTEQUALS   : '!=' ;
LPAREN      : '(' ;
RPAREN      : ')' ;
GREATER     : '>' ;
LESS        : '<' ;
GREATEREQ   : '>=' ;
LESSEQ      : '<=' ;
NOT         : '!' ;
OR          : O R;
AND         : A N D;

FLOATLIT    : '0.'[0-9]+|[1-9][0-9]* '.'[0-9]+;
INTLIT      : '0'|[1-9][0-9]*;
ID          : [_]*[A-Za-z0-9_][A-Za-z0-9_]* ;
BOOLEANLIT  : ('True'|'False');
STRINGLIT   : '"'ID'"';

fragment A : [aA];
fragment B : [bB];
fragment C : [cC];
fragment D : [dD];
fragment E : [eE];
fragment F : [fF];
fragment G : [gG];
fragment H : [hH];
fragment I : [iI];
fragment J : [jJ];
fragment K : [kK];
fragment L : [lL];
fragment M : [mM];
fragment N : [nN];
fragment O : [oO];
fragment P : [pP];
fragment Q : [qQ];
fragment R : [rR];
fragment S : [sS];
fragment T : [tT];
fragment U : [uU];
fragment V : [vV];
fragment W : [wW];
fragment X : [xX];
fragment Y : [yY];
fragment Z : [zZ];


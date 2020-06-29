lexer grammar GpuSqlLexer;

DATETIMELIT: SQOUTE DATELIT ([ ] TIMELIT)? SQOUTE;

LF: '\n' -> skip;
CR: '\r' -> skip;
CRLF: '\r\n' -> skip;
WS: [\t\r\n ]+ -> skip;
SEMICOL: ';';
SQOUTE: '\'';
DQOUTE: '"';
UNDERSCORE: '_';
COLON: ':';
COMMA: ',';
DOT: '.';
LSQR_BRC: '[';
RSQR_BRC: ']';
STRING: DQOUTE ( ~[\\"\r\n])* DQOUTE;
DELIMID: LSQR_BRC ( ~[\\[\]$\r\n\u002e])+ RSQR_BRC;

DATELIT: YEARLIT MINUS MONTHLIT MINUS DAYLIT;

fragment YEARLIT: [1-9][0-9][0-9][0-9];
fragment MONTHLIT: [1][0-2]| [0][1-9];
fragment DAYLIT: [0][1-9]| [1-2][0-9]| [3][0-1];

TIMELIT: HOURLIT COLON MINUTELIT COLON SECONDLIT;

fragment HOURLIT: [0-1][0-9]| [2][0-3];
fragment MINUTELIT: [0-5][0-9];
fragment SECONDLIT: [0-5][0-9];

POINT: 'POINT';
MULTIPOINT: 'MULTIPOINT';
LINESTRING: 'LINESTRING';
MULTILINESTRING: 'MULTILINESTRING';
POLYGON: 'POLYGON';
MULTIPOLYGON: 'MULTIPOLYGON';

INTTYPE: (I N T) | (I N T E G E R) | (I N T '3' '2');
LONGTYPE: (L O N G) | (I N T '6' '4');
DATETYPE: D A T E;
DETETIMETYPE: D A T E T I M E;
FLOATTYPE: F L O A T;
DOUBLETYPE: D O U B L E;
STRINGTYPE: S T R I N G;
BOOLEANTYPE: ((B O O L E A N) | (B O O L) | (I N T '8'));
POINTTYPE: G E O '_' P O I N T;
POLYTYPE: G E O '_' P O L Y G O N;
TYPES: T Y P E S;

INDEX: I N D E X;
UNIQUE: U N I Q U E;
PRIMARY: P R I M A R Y;
KEY: K E Y;

CREATE: C R E A T E;
ADD: A D D;
DROP: D R O P;
ALTER: A L T E R;
RENAME: R E N A M E;
SET: S E T;
INSERT: I N S E R T;
INTO: I N T O;
SHOW: S H O W;
SIZE: S I Z E;
FULL: F U L L;
OUTER: O U T E R;
INNER: I N N E R;

DATABASE: D A T A B A S E;
DATABASES: D A T A B A S E S;
TABLE: T A B L E;
TABLES: T A B L E S;
COLUMN: C O L U M N;
COLUMNS: C O L U M N S;
BLOCK: B L O C K;
CONSTRAINTS: C O N S T R A I N T S;

VALUES: V A L U E S;
SELECT: S E L E C T;
FROM: F R O M;
JOIN: J O I N;
WHERE: W H E R E;
GROUP: G R O U P;
AS: A S;
IN: I N;
TO: T O;
IS: I S;
NOT: N O T;
NULL_T: N U L L;
BY: B Y;
BETWEEN: B E T W E E N;
ON: O N;
ORDER: O R D E R;
DIR: (A S C) | (D E S C);
LIMIT: L I M I T;
OFFSET: O F F S E T;
QUERY: Q U E R Y;

AVG_AGG: A V G;
SUM_AGG: S U M;
MIN_AGG: M I N;
MAX_AGG: M A X;
COUNT_AGG: C O U N T;

YEAR: Y E A R;
MONTH: M O N T H;
DAY: D A Y;
HOUR: H O U R;
MINUTE: M I N U T E;
SECOND: S E C O N D;
WEEKDAY: W E E K D A Y;
DAYOFWEEK: D A Y O F W E E K;
NOW: N O W LPAREN RPAREN;

PI: P I LPAREN RPAREN;
ABS: A B S;
SIN: S I N;
COS: C O S;
TAN: T A N;
COT: C O T;
ASIN: A S I N;
ACOS: A C O S;
ATAN: A T A N;
ATAN2: A T A N '2';

LOG10: L O G '10';
LOG: L O G;
EXP: E X P;
POW: P O W;
SQRT: S Q R T;
SQUARE: S Q U A R E;
SIGN: S I G N;
ROOT: R O O T;
ROUND: R O U N D;
CEIL: C E I L;
FLOOR: F L O O R;

LTRIM: L T R I M;
RTRIM: R T R I M;
LOWER: L O W E R;
UPPER: U P P E R;
REVERSE: R E V E R S E;
LEN: L E N;
LEFT: L E F T;
RIGHT: R I G H T;
CONCAT: C O N C A T;

CAST: C A S T;
RETPAYLOAD: R E T P A Y L O A D;

GEO_CONTAINS: G E O UNDERSCORE C O N T A I N S;
GEO_INTERSECT: G E O UNDERSCORE I N T E R S E C T;
GEO_UNION: G E O UNDERSCORE U N I O N;

GEO_LONGITUDE_TO_TILE_X: G E O UNDERSCORE L O N G I T U D E UNDERSCORE T O UNDERSCORE T I L E UNDERSCORE X;
GEO_LATITUDE_TO_TILE_Y: G E O UNDERSCORE L A T I T U D E UNDERSCORE T O UNDERSCORE T I L E UNDERSCORE Y;
GEO_TILE_X_TO_LONGITUDE: G E O UNDERSCORE T I L E UNDERSCORE X UNDERSCORE T O UNDERSCORE L O N G I T U D E;
GEO_TILE_Y_TO_LATITUDE: G E O UNDERSCORE T I L E UNDERSCORE Y UNDERSCORE T O UNDERSCORE L A T I T U D E;

PLUS: '+';
MINUS: '-';
ASTERISK: '*';
DIVISION: '/';
MODULO: '%';
XOR: '^';
EQUALS: '=';
NOTEQUALS: '!=';
NOTEQUALS_GT_LT: '<>';
LPAREN: '(';
RPAREN: ')';
GREATER: '>';
LESS: '<';
GREATEREQ: '>=';
LESSEQ: '<=';
LOGICAL_NOT: '!';
OR: O R;
AND: A N D;
BIT_OR: '|';
BIT_AND: '&';
L_SHIFT: '<<';
R_SHIFT: '>>';

BOOLEANLIT: (TRUE | FALSE);
TRUE: T R U E;
FALSE: F A L S E;
FLOATLIT: ('0.' [0-9]+ | [1-9][0-9]* '.' [0-9]+);
INTLIT: ('0' | [1-9][0-9]*);
ID: [_]* [A-Za-z0-9_][A-Za-z0-9_]*;

fragment A: [aA];
fragment B: [bB];
fragment C: [cC];
fragment D: [dD];
fragment E: [eE];
fragment F: [fF];
fragment G: [gG];
fragment H: [hH];
fragment I: [iI];
fragment J: [jJ];
fragment K: [kK];
fragment L: [lL];
fragment M: [mM];
fragment N: [nN];
fragment O: [oO];
fragment P: [pP];
fragment Q: [qQ];
fragment R: [rR];
fragment S: [sS];
fragment T: [tT];
fragment U: [uU];
fragment V: [vV];
fragment W: [wW];
fragment X: [xX];
fragment Y: [yY];
fragment Z: [zZ];


#lang s-exp "type-env-lang.rkt"

(require "../types/abbrev.rkt" "../types/union.rkt"
         "../types/numeric-tower.rkt" "../rep/type-rep.rkt")

[Complex -Number]
[Number -Number]
[Inexact-Complex -InexactComplex]
[Single-Flonum-Complex -SingleFlonumComplex]
[Float-Complex -FloatComplex]
[Exact-Number -ExactNumber]
[Real -Real]
[Nonpositive-Real -NonPosReal]
[Negative-Real -NegReal]
[Nonnegative-Real -NonNegReal]
[Positive-Real -PosReal]
[Real-Zero -RealZero]
[Inexact-Real -InexactReal]
[Single-Flonum -SingleFlonum]
[Nonpositive-Inexact-Real -NonPosInexactReal]
[Nonpositive-Single-Flonum -NonPosSingleFlonum]
[Negative-Inexact-Real -NegInexactReal]
[Negative-Single-Flonum -NegSingleFlonum]
[Nonnegative-Inexact-Real -NonNegInexactReal]
[Nonnegative-Single-Flonum -NonNegSingleFlonum]
[Positive-Inexact-Real -PosInexactReal]
[Positive-Single-Flonum -PosSingleFlonum]
[Inexact-Real-Nan -InexactRealNan]
[Inexact-Real-Zero -InexactRealZero]
[Inexact-Real-Negative-Zero -InexactRealNegZero]
[Inexact-Real-Positive-Zero -InexactRealPosZero]
[Single-Flonum-Nan -SingleFlonumNan]
[Single-Flonum-Zero -SingleFlonumZero]
[Single-Flonum-Negative-Zero -SingleFlonumNegZero]
[Single-Flonum-Positive-Zero -SingleFlonumPosZero]
;; these are the default, 64-bit floats, can be optimized
[Float -Flonum] ; both of these are valid
[Flonum -Flonum]
[Nonpositive-Float -NonPosFlonum] ; both of these are valid
[Nonpositive-Flonum -NonPosFlonum]
[Negative-Float -NegFlonum] ; both of these are valid
[Negative-Flonum -NegFlonum]
[Nonnegative-Float -NonNegFlonum] ; both of these are valid
[Nonnegative-Flonum -NonNegFlonum]
[Positive-Float -PosFlonum] ; both of these are valid
[Positive-Flonum -PosFlonum]
[Float-Nan -FlonumNan]
[Flonum-Nan -FlonumNan]
[Float-Zero -FlonumZero] ; both of these are valid
[Flonum-Zero -FlonumZero]
[Float-Negative-Zero -FlonumNegZero] ; both of these are valid
[Flonum-Negative-Zero -FlonumNegZero]
[Float-Positive-Zero -FlonumPosZero] ; both of these are valid
[Flonum-Positive-Zero -FlonumPosZero]
[Exact-Rational -Rat]
[Nonpositive-Exact-Rational -NonPosRat]
[Negative-Exact-Rational -NegRat]
[Nonnegative-Exact-Rational -NonNegRat]
[Positive-Exact-Rational -PosRat]
[Integer -Int]
[Nonpositive-Integer -NonPosInt]
[Negative-Integer -NegInt]
[Exact-Nonnegative-Integer -Nat] ; all three of these are valid
[Nonnegative-Integer -Nat]
[Natural -Nat]
[Exact-Positive-Integer -PosInt] ; both of these are valid
[Positive-Integer -PosInt]
[Fixnum -Fixnum]
[Negative-Fixnum -NegFixnum]
[Nonpositive-Fixnum -NonPosFixnum]
[Nonnegative-Fixnum -NonNegFixnum]
[Positive-Fixnum -PosFixnum]
[Index -Index]
[Positive-Index -PosIndex]
[Byte -Byte]
[Positive-Byte -PosByte]
[Zero (-val 0)]
[One  (-val 1)]
[ExtFlonum -ExtFlonum]
[Nonpositive-ExtFlonum -NonPosExtFlonum]
[Negative-ExtFlonum -NegExtFlonum]
[Nonnegative-ExtFlonum -NonNegExtFlonum]
[Positive-ExtFlonum -PosExtFlonum]
[ExtFlonum-Nan -ExtFlonumNan]
[ExtFlonum-Zero -ExtFlonumZero]
[ExtFlonum-Negative-Zero -ExtFlonumNegZero]
[ExtFlonum-Positive-Zero -ExtFlonumPosZero]

[Void -Void]
[Undefined -Undefined] ; initial value of letrec bindings
[Boolean -Boolean]
[Symbol -Symbol]
[String -String]
[Any Univ]
[Port -Port]
[Path -Path]
[Path-For-Some-System -SomeSystemPath]
[Path-String -Pathlike]
[Regexp -Regexp]
[PRegexp -PRegexp]
[Byte-Regexp -Byte-Regexp]
[Byte-PRegexp -Byte-PRegexp]
[Char -Char]
[Namespace -Namespace]
[Input-Port -Input-Port]
[Output-Port -Output-Port]
[Bytes -Bytes]
[EOF (-val eof)]
[Sexpof (-poly (a) (-Sexpof a))]   ;; recursive union of sexps with a
[Syntaxof (-poly (a) (-Syntax a))] ;; syntax-e yields a
[Syntax-E In-Syntax] ;; possible results of syntax-e on "2D" syntax
[Syntax Any-Syntax]  ;; (Syntaxof Syntax-E): "2D" syntax
[Datum Syntax-Sexp]  ;; (Sexpof Syntax), datum->syntax yields "2D" syntax
[Sexp -Sexp]         ;; (Sexpof (U)), syntax->datum of "2D" syntax
[Identifier Ident]
[Procedure top-func]
[BoxTop -BoxTop]
[Weak-BoxTop -Weak-BoxTop]
[ChannelTop -ChannelTop]
[Async-ChannelTop -Async-ChannelTop]
[VectorTop -VectorTop]
[HashTableTop -HashTop]
[MPairTop -MPairTop]
[Thread-CellTop -Thread-CellTop]
[Prompt-TagTop -Prompt-TagTop]
[Continuation-Mark-KeyTop -Continuation-Mark-KeyTop]
[Struct-TypeTop (make-StructTypeTop)]
[ClassTop (make-ClassTop)]
[UnitTop (make-UnitTop)]
[Keyword -Keyword]
[Thread -Thread]
[Resolved-Module-Path -Resolved-Module-Path]
[Module-Path -Module-Path]
[Module-Path-Index -Module-Path-Index]
[Compiled-Module-Expression -Compiled-Module-Expression]
[Compiled-Expression -Compiled-Expression]
[Read-Table -Read-Table]
[Special-Comment -Special-Comment]
[Struct-Type-Property -Struct-Type-Property]
[Pretty-Print-Style-Table -Pretty-Print-Style-Table]
[UDP-Socket -UDP-Socket]
[TCP-Listener -TCP-Listener]
[Custodian -Custodian]
[Parameterization -Parameterization]
[Inspector -Inspector]
[Namespace-Anchor -Namespace-Anchor]
[Variable-Reference -Variable-Reference]
[Internal-Definition-Context -Internal-Definition-Context]
[Subprocess -Subprocess]
[Security-Guard -Security-Guard]
[Thread-Group -Thread-Group]
[Impersonator-Property -Impersonator-Property]
[Semaphore -Semaphore]
[FSemaphore -FSemaphore]
[Bytes-Converter -Bytes-Converter]
[Pseudo-Random-Generator -Pseudo-Random-Generator]
[Logger -Logger]
[Log-Receiver -Log-Receiver]
[Log-Level -Log-Level]
[Place-Channel -Place-Channel]
[Place -Place]
[Will-Executor -Will-Executor]


[Listof -Listof]
[Vectorof (-poly (a) (make-Vector a))]
[FlVector -FlVector]
[ExtFlVector -ExtFlVector]
[FxVector -FxVector]
[Option (-poly (a) (-opt a))]
[HashTable (-poly (a b) (-HT a b))]
[Promise (-poly (a) (-Promise a))]
[Pair (-poly (a b) (-pair a b))]
[Boxof (-poly (a) (make-Box a))]
[Weak-Boxof (-poly (a) (-weak-box a))]
[Channelof (-poly (a) (make-Channel a))]
[Async-Channelof (-poly (a) (make-Async-Channel a))]
[Ephemeronof (-poly (a) (make-Ephemeron a))]
[Setof (-poly (e) (make-Set e))]
[Evtof (-poly (r) (-evt r))]
[Continuation-Mark-Set -Cont-Mark-Set]
[False -False]
[True -True]
[Null -Null]
[Nothing (Un)]
[Futureof (-poly (a) (-future a))]
[Pairof (-poly (a b) (-pair a b))]
[MPairof (-poly (a b) (-mpair a b))]
[MListof (-poly (a) (-mlst a))]
[Thread-Cellof (-poly (a) (-thread-cell a))]
[Custodian-Boxof (-poly (a) (make-CustodianBox a))]

[Continuation-Mark-Keyof (-poly (a) (make-Continuation-Mark-Keyof a))]
[Prompt-Tagof (-poly (a b) (make-Prompt-Tagof a b))]

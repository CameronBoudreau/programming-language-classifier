;;; Copyright (C) 2008 by Sam Steingold
;;; This is Free Software, covered by the GNU GPL (v2)
;;; See http://www.gnu.org/copyleft/gpl.html
;;;
;;; Study which functions should be added to eval.d:FUNTAB.
;;; This is not TRT, of course: we are counting how many other functions
;;; call the not-inlined system functions, not how often they are actually
;;; called, but this still may be useful.

(defun byte-compiled-closure (fdef)
  "Check whether the object is a byte compiled lisp function."
  (and (sys::closurep fdef) (compiled-function-p fdef)
       (not (sys::subr-info fdef))
       fdef))

(defparameter *closures*
  (let ((ht (make-hash-table)))
    (do-all-symbols (s)
      (when (and (fboundp s) (not (special-operator-p s))
                 (byte-compiled-closure (fdefinition s)))
        (setf (gethash s ht) t)))
    ht)
  "All compiled Lisp closure names.")

(defparameter *interpreted-functions* ;empty!
  (let ((ht (make-hash-table)))
    (do-all-symbols (s)
      (when (and (fboundp s) (not (special-operator-p s))
                 (let ((fdef (fdefinition s)))
                   (and (sys::closurep fdef) (not (compiled-function-p fdef))
                        (not (typep fdef 'generic-function)))))
        (setf (gethash s ht) t)))
    ht)
  "All compiled Lisp closure names.")

(defparameter *notinline-subrs*
  (let ((ht (make-hash-table)))
    (dolist (s (nth-value 2 (module-info "clisp" t)) ht)
      ;; some subrs, e.g., LOAD, are later redefined in Lisp
      (when (and (sys::subr-info s) (not (gethash s sys::function-codes)))
        (setf (gethash s ht) (cons 0 nil)))))
  "Lisp functions implemented in C but not inlined in FUNTAB.
Values are pairs of (caller-count . caller-list),
i.e. (car pair) = (length (cdr pair))")

(defun incf-notinline-subr (caller s)
  "Augment *NOTINLINE-SUBRS* for the given S called by CALLER."
  (let ((pair (gethash s *notinline-subrs*)))
    (when pair
      (incf (car pair))
      (push caller (cdr pair)))))

(defun show-count-ht (ht)
  "Print map (object -> (count . callers) nicely, dropping 0 counts."
  (loop :with *print-length* = 5 :and *print-pretty* = nil
    :for (sym count . callers)
    :in (sort (ext:with-collect (c)
                (maphash (lambda (sym count-callers)
                           (when (cdr count-callers)
                             (c (cons sym count-callers))))
                         ht))
              #'> :key #'second)
    :do (format t "~10:D  ~S  ~S~%" count sym callers)))

(defun map-cclosures (f)
  "Call f on all lisp compiled closures once."
  (let ((done (make-hash-table)))
    (labels ((mapsubs (closure)
               (setf (gethash closure done) t)
               (funcall f closure)
               (dolist (const (sys::closure-consts closure))
                 (when (and (byte-compiled-closure const)
                            (not (gethash const done)))
                   (mapsubs const)))))
      (maphash (lambda (sym _t)
                 (declare (ignore _t))
                 (mapsubs (fdefinition sym)))
             *closures*))))

;; map-calls is by far the trickiest part of the code
(defgeneric map-calls (f closure)
  (:documentation "Call F on each function call in the CLOSURE.")
  (:method ((f t) (closure function))
    (multiple-value-bind (req-num opt-num rest-p
                          key-p keyword-list allow-other-keys-p
                          byte-list const-list)
        (sys::signature closure)
      (declare (ignore req-num opt-num rest-p
                       key-p keyword-list allow-other-keys-p))
      (let ((lap-list (sys::disassemble-LAP byte-list const-list))
            (caller (sys::closure-name closure)))
        (dolist (L lap-list)
          (let* ((instr (cdr L))
                 (const (when (consp instr)
                          (case (first instr)
                            ((SYS::CALL SYS::CALL&PUSH)
                             (nth (third instr) const-list))
                            ((SYS::CALL0 SYS::CALL1 SYS::CALL1&PUSH
                              SYS::CALL1&JMPIFNOT SYS::CALL1&JMPIF
                              SYS::CALL2 SYS::CALL2&PUSH SYS::CALL2&JMPIFNOT
                              SYS::CALL2&JMPIF
                              SYS::CONST&SYMBOL-FUNCTION&PUSH
                              SYS::CONST&SYMBOL-FUNCTION
                              SYS::COPY-CLOSURE&PUSH SYS::COPY-CLOSURE
                              SYS::CONST&SYMBOL-FUNCTION&STORE
                              SYS::TAGBODY-OPEN SYS::HANDLER-OPEN)
                             (nth (second instr) const-list))))))
            (when (and const (symbolp const))
              (funcall f caller const)))))))
  (:method ((f t) (gf generic-function))
    (dolist (m (generic-function-methods gf))
      (map-calls f (clos::std-method-fast-function m)))))

#+(or) (progn

(map-cclosures (lambda (closure) (map-calls #'incf-notinline-subr closure)))

(show-count-ht *notinline-subrs*)

;;; 2008-03-01
       691  SYSTEM::TEXT  (REGEXP:REGEXP-SPLIT POSIX::MAKE-XTERM-IO-STREAM-1-1 SET-PPRINT-DISPATCH ED TYPEP ...)
       155  CLOS::TYPEP-CLASS  (TYPEP DESCRIBE ENSURE-GENERIC-FUNCTION ENSURE-GENERIC-FUNCTION SYSTEM::WARN-OF-TYPE ...)
        63  CLOS::DEFINED-CLASS-P  (TYPEP SYSTEM::SUBTYPE-SEQUENCE SYSTEM::CLOS-CLASS SYSTEM::CANONICALIZE-TYPE SYSTEM::FIND-STRUCTURE-SLOT-INITFUNCTION ...)
        37  CLOS::|(SETF STANDARD-INSTANCE-ACCESS)|  ((SETF CLOS::METHOD-COMBINATION-OPTIONS) (SETF CLOS::STD-GF-INITIALIZED) (SETF CLOS::STD-METHOD-GENERIC-FUNCTION) (SETF CLOS::STD-METHOD-FUNCTION) (SETF CLOS::STD-GF-LAMBDA-LIST) ...)
        36  STANDARD-INSTANCE-ACCESS  (CLOS::STD-GF-DOCUMENTATION CLOS::STD-METHOD-SPECIALIZERS CLOS::METHOD-COMBINATION-CHECK-METHOD-QUALIFIERS CLOS::METHOD-COMBINATION-IDENTITY-WITH-ONE-ARGUMENT CLOS::STD-GF-INITIALIZED ...)
        23  PROPER-LIST-P  (CLOS::COMPUTE-SLOTS-<SLOTTED-CLASS>-AROUND CLOS::SHARED-INITIALIZE-<DEFINED-CLASS> CLOS::SHARED-INITIALIZE-<DEFINED-CLASS> CLOS::SHARED-INITIALIZE-<DEFINED-CLASS> CLOS::COMPUTE-EFFECTIVE-SLOT-DEFINITION-INITARGS-<DEFINED-CLASS> ...)
        19  MAPCAP  (SYSTEM::C-LABELS SYSTEM::WRAP-USER-COMMANDS SYSTEM::C-FUNCTION-MACRO-LET SYSTEM::C-FLET SYSTEM::C-GENERIC-FLET ...)
        15  GETENV  (SHORT-SITE-NAME LONG-SITE-NAME LOAD-LOGICAL-PATHNAME-TRANSLATIONS LOAD-LOGICAL-PATHNAME-TRANSLATIONS LOAD-LOGICAL-PATHNAME-TRANSLATIONS ...)
        11  SYSTEM::|(SETF PACKAGE-LOCK)|  (SYSTEM::C-WITHOUT-PACKAGE-LOCK SYSTEM::C-WITHOUT-PACKAGE-LOCK CLOS::SET-<FORWARD-REFERENCED-CLASS>-<MISDESIGNED-FORWARD-REFERENCED-CLASS> CLOS::SET-<FORWARD-REFERENCED-CLASS>-<MISDESIGNED-FORWARD-REFERENCED-CLASS> CLOS::SET-<CLASS>-<POTENTIAL-CLASS> ...)
        11  SYSTEM::MACROP  (COMPILE SYSTEM::%EXPAND-FORM SYSTEM::%EXPAND-FORM SYSTEM::UNWRAPPED-FDEFINITION SYSTEM::TRACE1 ...)
         9  SYSTEM::ENCODINGP  (TYPEP SYSTEM::SIMPLIFY-AND-CHARACTER-1 SYSTEM::CANONICALIZE-TYPE SYSTEM::SUBTYPEP-CHARACTER SYSTEM::SUBTYPEP-CHARACTER ...)
         9  SYSTEM::%COMPILED-FUNCTION-P  (FUNCTION-LAMBDA-EXPRESSION ED COMPILE COMPILE SYSTEM::FUNCTION-SIGNATURE ...)
         8  SYSTEM::FORMAT-TABULATE  (PPRINT-TAB COMMON-LISP::APROPOS-1 SYSTEM::DESCRIBE-SLOTTED-OBJECT-1 SYSTEM::DESCRIBE-SLOTTED-OBJECT-1 SYSTEM::BREAK-LOOP ...)
         7  PPRINT-NEWLINE  (COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-TABULAR-1 ...)
         7  SYSTEM::WRITE-UNREADABLE  (CLOS::PRINT-OBJECT-<POTENTIAL-CLASS> CLOS::PRINT-OBJECT-<FORWARD-REFERENCE-TO-CLASS> CLOS::PRINT-OBJECT-<STANDARD-METHOD> CLOS::PRINT-OBJECT-<SLOT-DEFINITION> CLOS::PRINT-OBJECT-<EQL-SPECIALIZER> ...)
         6  STRING-WIDTH  (SYSTEM::WARN-OF-TYPE SYSTEM::FILL-STREAM-LINE-POSITION SYSTEM::WRITE-TO-SHORT-STRING SYSTEM::WRITE-TO-SHORT-STRING SYSTEM::FORMAT-JUSTIFIED-SEGMENTS ...)
         6  SYSTEM::WRITE-SPACES  (SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::STREAM-TAB SYSTEM::TRACE-OUTPUT ...)
         6  SHELL  (MAKE-XTERM-IO-STREAM SYSTEM::DISASSEMBLE-MACHINE-CODE SYSTEM::DISASSEMBLE-MACHINE-CODE SYSTEM::DISASSEMBLE-MACHINE-CODE EDIT-FILE ...)
         6  SYSTEM::SYMBOL-MACRO-P  (SYSTEM::%EXPAND-VARLIST-MACROP SYSTEM::%EXPAND-FORM SYSTEM::VENV-SEARCH-MACRO SYSTEM::%EXPAND-SETQLIST-MACROP SYSTEM::VENV-SEARCH ...)
         6  SYSTEM::MACRO-EXPANDER  (COMPILE SYSTEM::%EXPAND-FORM SYSTEM::UNWRAPPED-FDEFINITION SYSTEM::FENV-SEARCH SYSTEM::FENV-SEARCH ...)
         6  SYSTEM::%UNBOUND  (CLOS::PRINT-OBJECT-<STANDARD-METHOD>-1 CLOS::PRINT-OBJECT-<STANDARD-METHOD>-1 (SETF CLOS::CLASS-VALID-INITARGS-FROM-SLOTS) CLOS::CREATE-SHARED-SLOTS-VECTOR CLOS::STD-GF-UNDETERMINEDP ...)
         5  SYSTEM::CHECK-FUNCTION-NAME  ((SETF COMPILER-MACRO-FUNCTION) COMPILER-MACRO-FUNCTION COMPILE SYSTEM::CHECK-TRACEABLE CLOS::ANALYZE-DEFGENERIC)
         5  SYSTEM::FRAME-UP  (SYSTEM::FRAME-UP-DOWN SYSTEM::DEBUG-TOP SYSTEM::FRAME-LIMIT-UP SYSTEM::DEBUG-UP SYSTEM::FRAME-LIMIT-DOWN)
         5  SYSTEM::DESCRIBE-FRAME  (SYSTEM::DEBUG-TOP SYSTEM::DEBUG-UP SYSTEM::DEBUG-WHERE SYSTEM::DEBUG-BOTTOM SYSTEM::DEBUG-DOWN)
         5  ENCODING-CHARSET  (SYSTEM::SIMPLIFY-AND-CHARACTER-1 SYSTEM::ENCODING-ZEROES SYSTEM::SUBTYPEP-CHARACTER SYSTEM::SUBTYPEP-CHARACTER SYSTEM::SUBTYPEP-CHARACTER-PRE-SIMPLIFY)
         5  SYSTEM::EXPAND-DEFTYPE  (TYPEP SYSTEM::SUBTYPE-SEQUENCE SYSTEM::CANONICALIZE-TYPE SYSTEM::SUBTYPE-INTEGER TYPE-EXPAND)
         4  SYSTEM::FRAME-DOWN  (SYSTEM::FRAME-UP-DOWN SYSTEM::FRAME-LIMIT-UP SYSTEM::DEBUG-BOTTOM SYSTEM::DEBUG-DOWN)
         4  CONVERT-STRING-TO-BYTES  (SYSTEM::HTTP-ERROR SYSTEM::ENCODING-ZEROES SYSTEM::ENCODING-ZEROES OPEN-HTTP)
         4  SIGNAL  (CERROR SYSTEM::CHECK-TYPE-FAILED-4-1 SYSTEM::WARN-OF-TYPE SYSTEM::COERCE-TO-CONDITION)
         4  PPRINT-INDENT  (COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-TABULAR-1 SYSTEM::FORMAT-PPRINT-INDENT)
         4  CLOS::POTENTIAL-CLASS-P  (CLOS::CLASS-CLASSNAME (SETF CLOS::CLASS-DIRECT-SUBCLASSES-TABLE) CLOS::CLASS-DIRECT-SUBCLASSES-TABLE (SETF CLOS::CLASS-CLASSNAME))
         4  SYSTEM::MAKE-MACRO  (COMPILE SYSTEM::TRACE1 SYSTEM::MAKE-MACRO-EXPANDER SYSTEM::MAKE-FUNMACRO-EXPANDER)
         4  SYSTEM::LINE-NUMBER  (COMPILE-FILE COMPILE-FILE LOAD LOAD)
         4  SYSTEM::%PUTF  (SYSTEM::%SET-DOCUMENTATION SYSTEM::SET-FILE-DOC CLOS::SET-FUNCTION-DOCUMENTATION CLOS::ENSURE-GENERIC-FUNCTION-USING-CLASS-<T>)
         3  SYSTEM::ADD-IMPLICIT-BLOCK  (SYSTEM::C-LABELS SYSTEM::C-FLET SYSTEM::%EXPAND-LAMBDABODY)
         3  STRING-INVERTCASE  (SYSTEM::COMPLETION-1 SYSTEM::COMPLETION SYSTEM::COMPLETION)
         3  SYSTEM::READ-EVAL-PRINT  (SYSTEM::STEP-HOOK-FN-1-3 SYSTEM::BREAK-LOOP-2-3 SYSTEM::MAIN-LOOP-1)
         3  SYSTEM::%PPRINT-LOGICAL-BLOCK  (PPRINT-LINEAR PPRINT-FILL PPRINT-TABULAR)
         3  SYSTEM::%CIRCLEP  (COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-TABULAR-1)
         3  PACKAGE-CASE-INVERTED-P  (SYSTEM::MODULE-NAME SYSTEM::COMPLETION FFI::TO-C-NAME)
         3  SYSTEM::HEAP-STATISTICS  (ROOM SYSTEM::%SPACE1 SYSTEM::%SPACE)
         3  SYMBOL-MACRO-EXPAND  (APROPOS SYSTEM::VENV-SEARCH-MACRO SYSTEM::VENV-ASSOC)
         3  SYSTEM::MACRO-LAMBDA-LIST  (COMPILE SYSTEM::TRACE1 ARGLIST)
         3  SYSTEM::FUNCTION-MACRO-P  (SYSTEM::%EXPAND-FORM SYSTEM::%EXPAND-FORM SYSTEM::FENV-SEARCH)
         3  WEAK-LIST-P  (CLOS::REMOVE-FROM-WEAK-SET CLOS::ADD-TO-WEAK-SET CLOS::LIST-WEAK-SET)
         3  WEAK-LIST-LIST  (CLOS::REMOVE-FROM-WEAK-SET CLOS::ADD-TO-WEAK-SET CLOS::LIST-WEAK-SET)
         3  WRITE-CHAR-SEQUENCE  (SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER)
         3  MAKE-PIPE-INPUT-STREAM  (SHORT-SITE-NAME LONG-SITE-NAME RUN-SHELL-COMMAND)
         3  SYSTEM::%REMF  (SYSTEM::%SET-DOCUMENTATION CLOS::ENSURE-GENERIC-FUNCTION-USING-CLASS-<T> CLOS::ENSURE-CLASS-USING-CLASS-<T>)
         2  SYSTEM::FUNCTION-BLOCK-NAME  (SYSTEM::MAKE-MACRO-EXPANSION CLOS::ANALYZE-METHOD-DESCRIPTION)
         2  NSTRING-INVERTCASE  (SYSTEM::COMPLETION-4 SYSTEM::COMPLETION)
         2  SYSTEM::READ-FORM  (SYSTEM::STEP-HOOK-FN SYSTEM::DEBUG-RETURN)
         2  SYSTEM::THE-FRAME  (SYSTEM::FRAME-LIMIT-UP SYSTEM::FRAME-LIMIT-DOWN)
         2  SYSTEM::SAME-ENV-AS  (SYSTEM::STEP-HOOK-FN-1 SYSTEM::BREAK-LOOP-2)
         2  SYSTEM::TRAP-EVAL-FRAME  (SYSTEM::DEBUG-TRAP-OFF SYSTEM::DEBUG-TRAP-ON)
         2  GC  (SYSTEM::%SPACE2 SYSTEM::%SPACE1)
         2  READTABLE-CASE  (SYSTEM::COMPLETION SYSTEM::COMPLETION)
         2  LIST-LENGTH-DOTTED  (SYSTEM::DESTRUCTURING-ERROR SYSTEM::MACRO-CALL-ERROR)
         2  LIST-LENGTH-PROPER  (SYSTEM::LIST-TO-HT SYSTEM::LIST-TO-HT)
         2  SYSTEM::LIST-LENGTH-IN-BOUNDS-P  (SYSTEM::FINALIZE-COUTPUT-FILE-9 SYSTEM::CORRECTABLE-ERROR)
         2  SYSTEM::VERSION  (COMPILE-FILE SYSTEM::OPEN-FOR-LOAD-CHECK-COMPILED-FILE)
         2  SYSTEM::DEFAULT-TIME-ZONE  (DECODE-UNIVERSAL-TIME ENCODE-UNIVERSAL-TIME)
         2  SYSTEM::LIB-DIRECTORY  (REQUIRE SYSTEM::CLISP-DATA-FILE)
         2  SYSTEM::GC-STATISTICS  (SYSTEM::%SPACE2 SYSTEM::%SPACE1)
         2  SYSTEM::CLOSURE-CONST  (SYSTEM::%LOCAL-GET SYSTEM::PASS3)
         2  SYSTEM::SET-CLOSURE-CONST  (SYSTEM::PASS3 SYSTEM::%LOCAL-SET)
         2  SYSTEM::CLOSURE-SET-DOCUMENTATION  (COMPILE CLOS::SET-FUNCTION-DOCUMENTATION)
         2  SET-FUNCALLABLE-INSTANCE-FUNCTION  (CLOS::FINALIZE-FAST-GF CLOS::INSTALL-DISPATCH)
         2  SYSTEM::GENERIC-FUNCTION-EFFECTIVE-METHOD-FUNCTION  (CLOS::%CALL-NEXT-METHOD CLOS::%CALL-NEXT-METHOD)
         2  SYSTEM::MAKE-SYMBOL-MACRO  (SYSTEM::%EXPAND-FORM SYSTEM::C-SYMBOL-MACROLET)
         2  SYSTEM::|(SETF WEAK-LIST-LIST)|  (CLOS::REMOVE-FROM-WEAK-SET CLOS::ADD-TO-WEAK-SET)
         2  SYNONYM-STREAM-SYMBOL  (SYSTEM::STREAM-OUTPUT-ELEMENT-TYPE SYSTEM::STREAM-INPUT-ELEMENT-TYPE)
         2  TWO-WAY-STREAM-INPUT-STREAM  (SYSTEM::STREAM-INPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  TWO-WAY-STREAM-OUTPUT-STREAM  (SYSTEM::STREAM-OUTPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  ECHO-STREAM-INPUT-STREAM  (SYSTEM::STREAM-INPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  ECHO-STREAM-OUTPUT-STREAM  (SYSTEM::STREAM-OUTPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  SYSTEM::TERMINAL-RAW  (SYSTEM::EXEC-WITH-KEYBOARD SYSTEM::EXEC-WITH-KEYBOARD)
         2  MAKE-PIPE-OUTPUT-STREAM  (SYSTEM::MAKE-PRINTER-STREAM RUN-SHELL-COMMAND)
         2  INTERACTIVE-STREAM-P  (CERROR SYSTEM::BREAK-LOOP)
         2  SYSTEM::RANDOM-POSFIXNUM  (CLOS::SHARED-INITIALIZE-<STANDARD-STABLEHASH> CLOS::MAKE-STRUCTURE-STABLEHASH)
         1  SYSTEM::CHECK-SYMBOL  (CLOS::SHARED-INITIALIZE-<POTENTIAL-CLASS>)
         1  SYSTEM::EVAL-AT  (SYSTEM::STEP-HOOK-FN)
         1  SYSTEM::EVAL-FRAME-P  (SYSTEM::COMMANDS)
         1  SYSTEM::DRIVER-FRAME-P  (SYSTEM::FRAME-LIMIT-UP)
         1  SYSTEM::REDO-EVAL-FRAME  (SYSTEM::DEBUG-REDO)
         1  SYSTEM::RETURN-FROM-EVAL-FRAME  (SYSTEM::DEBUG-RETURN)
         1  SHOW-STACK  (SYSTEM::PRINT-BACKTRACE)
         1  SYSTEM::%ROOM  (ROOM)
         1  MAKE-ENCODING  (SYSTEM::GET-CHARSET-RANGE)
         1  SYSTEM::CHARSET-TYPEP  (TYPEP)
         1  SYSTEM::CHARSET-RANGE  (SYSTEM::GET-CHARSET-RANGE)
         1  SYSTEM::FOREIGN-ENCODING  (FFI::EXEC-WITH-FOREIGN-STRING)
         1  CONVERT-STRING-FROM-BYTES  (OPEN-HTTP)
         1  SYSTEM::CONSES-P  (SYSTEM::DS-TYPEP)
         1  SYSTEM::CURRENT-LANGUAGE  (LOCALIZED)
         1  SYSTEM::DELTA4  (SYSTEM::%TIME)
         1  PACKAGE-LOCK  (SYSTEM::F-SIDE-EFFECT)
         1  PACKAGE-CASE-SENSITIVE-P  (SYSTEM::COMPLETION)
         1  SYSTEM::SYMBOL-VALUE-LOCK  (SYSTEM::SET-CHECK-LOCK)
         1  SYSTEM::CHECK-PACKAGE-LOCK  (SYSTEM::CHECK-REDEFINITION)
         1  DELETE-PACKAGE  (INSPECT)
         1  SYSTEM::PACKAGE-ITERATOR  (SYSTEM::PACKAGE-ITERATOR-FUNCTION)
         1  SYSTEM::PACKAGE-ITERATE  (SYSTEM::PACKAGE-ITERATOR-FUNCTION-1)
         1  LOGICAL-PATHNAME  (SYSTEM::VALID-LOGICAL-PATHNAME-STRING-P-3)
         1  TRANSLATE-LOGICAL-PATHNAME  (SYSTEM::OPEN-FOR-LOAD)
         1  SYSTEM::MAKE-LOGICAL-PATHNAME  (SYSTEM::SET-LOGICAL-PATHNAME-TRANSLATIONS)
         1  USER-HOMEDIR-PATHNAME  (EDITOR-TEMPFILE)
         1  ABSOLUTE-PATHNAME  (SYSTEM::XSTRING)
         1  SYSTEM::SET-LIB-DIRECTORY  (SYSTEM::CLISP-DATA-FILE-1)
         1  SYSTEM::NOTE-NEW-STRUCTURE-CLASS  (CLOS::SHARED-INITIALIZE-<STRUCTURE-CLASS>)
         1  SYSTEM::NOTE-NEW-STANDARD-CLASS  (CLOS::FINALIZE-INSTANCE-SEMI-STANDARD-CLASS)
         1  SYSTEM::LIST-STATISTICS  (SYSTEM::%SPACE)
         1  SYSTEM::HEAP-STATISTICS-STATISTICS  (SYSTEM::%SPACE)
         1  SYSTEM::GC-STATISTICS-STATISTICS  (SYSTEM::%SPACE)
         1  SYSTEM::|(SETF CLOSURE-NAME)|  (SYSTEM::MAKE-PRELIMINARY)
         1  SYSTEM::CONSTANT-INITFUNCTION-P  (SYSTEM::DS-ARG-DEFAULT)
         1  SYSTEM::CLOSURE-DOCUMENTATION  (CLOS::FUNCTION-DOCUMENTATION)
         1  SYSTEM::CLOSURE-LAMBDA-LIST  (ARGLIST)
         1  SYSTEM::GLOBAL-SYMBOL-MACRO-DEFINITION  (SYSTEM::VENV-ASSOC)
         1  SYSTEM::FUNCTION-MACRO-EXPANDER  (SYSTEM::FENV-SEARCH)
         1  FINALIZE  (MAKE-XTERM-IO-STREAM)
         1  CLOS::ALLOCATE-METAOBJECT-INSTANCE  (CLOS::COPY-STANDARD-CLASS)
         1  CLOS::%CHANGE-CLASS  (CLOS::DO-CHANGE-CLASS)
         1  MAKE-WEAK-LIST  (CLOS::ADD-TO-WEAK-SET)
         1  WRITE-BYTE-SEQUENCE  (SYSTEM::HTTP-ERROR)
         1  BROADCAST-STREAM-STREAMS  (DRIBBLE-STREAM)
         1  SYSTEM::MAKE-KEYBOARD-STREAM  (SYSTEM::EXEC-WITH-KEYBOARD)
         1  MAKE-PIPE-IO-STREAM  (RUN-SHELL-COMMAND)
         1  SOCKET-ACCEPT  (SYSTEM::HTTP-COMMAND)
         1  SOCKET-CONNECT  (EXT::OPEN-HTTP-3)
         1  SYSTEM::SET-STREAM-EXTERNAL-FORMAT  (SYSTEM::SET-OUTPUT-STREAM-FASL)
         1  SYSTEM::STREAM-FASL-P  (SYSTEM::SET-OUTPUT-STREAM-FASL)
         1  FFI:VALIDP  (REGEXP::%MATCH)
         1  FFI::CALL-WITH-FOREIGN-STRING  (FFI::EXEC-WITH-FOREIGN-STRING)

;;; 2008-03-06
       692  SYSTEM::TEXT  (REGEXP:REGEXP-SPLIT POSIX::MAKE-XTERM-IO-STREAM-1-1 SET-PPRINT-DISPATCH ED TYPEP ...)
        15  GETENV  (SHORT-SITE-NAME LONG-SITE-NAME LOAD-LOGICAL-PATHNAME-TRANSLATIONS LOAD-LOGICAL-PATHNAME-TRANSLATIONS LOAD-LOGICAL-PATHNAME-TRANSLATIONS ...)
        11  SYSTEM::|(SETF PACKAGE-LOCK)|  (SYSTEM::C-WITHOUT-PACKAGE-LOCK SYSTEM::C-WITHOUT-PACKAGE-LOCK CLOS::SET-<FORWARD-REFERENCED-CLASS>-<MISDESIGNED-FORWARD-REFERENCED-CLASS> CLOS::SET-<FORWARD-REFERENCED-CLASS>-<MISDESIGNED-FORWARD-REFERENCED-CLASS> CLOS::SET-<CLASS>-<POTENTIAL-CLASS> ...)
         8  SYSTEM::FORMAT-TABULATE  (PPRINT-TAB COMMON-LISP::APROPOS-1 SYSTEM::DESCRIBE-SLOTTED-OBJECT-1 SYSTEM::DESCRIBE-SLOTTED-OBJECT-1 SYSTEM::BREAK-LOOP ...)
         7  PPRINT-NEWLINE  (COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-TABULAR-1 ...)
         7  SYSTEM::WRITE-UNREADABLE  (CLOS::PRINT-OBJECT-<POTENTIAL-CLASS> CLOS::PRINT-OBJECT-<FORWARD-REFERENCE-TO-CLASS> CLOS::PRINT-OBJECT-<STANDARD-METHOD> CLOS::PRINT-OBJECT-<SLOT-DEFINITION> CLOS::PRINT-OBJECT-<EQL-SPECIALIZER> ...)
         6  STRING-WIDTH  (SYSTEM::WARN-OF-TYPE SYSTEM::FILL-STREAM-LINE-POSITION SYSTEM::WRITE-TO-SHORT-STRING SYSTEM::WRITE-TO-SHORT-STRING SYSTEM::FORMAT-JUSTIFIED-SEGMENTS ...)
         6  SYSTEM::WRITE-SPACES  (SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::STREAM-TAB SYSTEM::TRACE-OUTPUT ...)
         6  SHELL  (MAKE-XTERM-IO-STREAM SYSTEM::DISASSEMBLE-MACHINE-CODE SYSTEM::DISASSEMBLE-MACHINE-CODE SYSTEM::DISASSEMBLE-MACHINE-CODE EDIT-FILE ...)
         5  SYSTEM::FRAME-UP  (SYSTEM::FRAME-UP-DOWN SYSTEM::DEBUG-TOP SYSTEM::FRAME-LIMIT-UP SYSTEM::DEBUG-UP SYSTEM::FRAME-LIMIT-DOWN)
         5  SYSTEM::DESCRIBE-FRAME  (SYSTEM::DEBUG-TOP SYSTEM::DEBUG-UP SYSTEM::DEBUG-WHERE SYSTEM::DEBUG-BOTTOM SYSTEM::DEBUG-DOWN)
         5  ENCODING-CHARSET  (SYSTEM::SIMPLIFY-AND-CHARACTER-1 SYSTEM::ENCODING-ZEROES SYSTEM::SUBTYPEP-CHARACTER SYSTEM::SUBTYPEP-CHARACTER SYSTEM::SUBTYPEP-CHARACTER-PRE-SIMPLIFY)
         5  SYSTEM::EXPAND-DEFTYPE  (TYPEP SYSTEM::SUBTYPE-SEQUENCE SYSTEM::CANONICALIZE-TYPE SYSTEM::SUBTYPE-INTEGER TYPE-EXPAND)
         4  SYSTEM::FRAME-DOWN  (SYSTEM::FRAME-UP-DOWN SYSTEM::FRAME-LIMIT-UP SYSTEM::DEBUG-BOTTOM SYSTEM::DEBUG-DOWN)
         4  CONVERT-STRING-TO-BYTES  (SYSTEM::HTTP-ERROR SYSTEM::ENCODING-ZEROES SYSTEM::ENCODING-ZEROES OPEN-HTTP)
         4  SIGNAL  (CERROR SYSTEM::CHECK-TYPE-FAILED-4-1 SYSTEM::WARN-OF-TYPE SYSTEM::COERCE-TO-CONDITION)
         4  PPRINT-INDENT  (COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-TABULAR-1 SYSTEM::FORMAT-PPRINT-INDENT)
         4  CLOS::POTENTIAL-CLASS-P  (CLOS::CLASS-CLASSNAME (SETF CLOS::CLASS-DIRECT-SUBCLASSES-TABLE) CLOS::CLASS-DIRECT-SUBCLASSES-TABLE (SETF CLOS::CLASS-CLASSNAME))
         4  SYSTEM::LINE-NUMBER  (COMPILE-FILE COMPILE-FILE LOAD LOAD)
         4  SYSTEM::%PUTF  (SYSTEM::%SET-DOCUMENTATION SYSTEM::SET-FILE-DOC CLOS::SET-FUNCTION-DOCUMENTATION CLOS::ENSURE-GENERIC-FUNCTION-USING-CLASS-<T>)
         3  SYSTEM::ADD-IMPLICIT-BLOCK  (SYSTEM::C-LABELS SYSTEM::C-FLET SYSTEM::%EXPAND-LAMBDABODY)
         3  STRING-INVERTCASE  (SYSTEM::COMPLETION-1 SYSTEM::COMPLETION SYSTEM::COMPLETION)
         3  SYSTEM::READ-EVAL-PRINT  (SYSTEM::STEP-HOOK-FN-1-3 SYSTEM::BREAK-LOOP-2-3 SYSTEM::MAIN-LOOP-1)
         3  SYSTEM::%PPRINT-LOGICAL-BLOCK  (PPRINT-LINEAR PPRINT-FILL PPRINT-TABULAR)
         3  SYSTEM::%CIRCLEP  (COMMON-LISP::PPRINT-LINEAR-1 COMMON-LISP::PPRINT-FILL-1 COMMON-LISP::PPRINT-TABULAR-1)
         3  PACKAGE-CASE-INVERTED-P  (SYSTEM::MODULE-NAME SYSTEM::COMPLETION FFI::TO-C-NAME)
         3  SYSTEM::HEAP-STATISTICS  (ROOM SYSTEM::%SPACE1 SYSTEM::%SPACE)
         3  SYMBOL-MACRO-EXPAND  (APROPOS SYSTEM::VENV-SEARCH-MACRO SYSTEM::VENV-ASSOC)
         3  SYSTEM::MACRO-LAMBDA-LIST  (COMPILE SYSTEM::TRACE1 ARGLIST)
         3  SYSTEM::FUNCTION-MACRO-P  (SYSTEM::%EXPAND-FORM SYSTEM::%EXPAND-FORM SYSTEM::FENV-SEARCH)
         3  WEAK-LIST-P  (CLOS::REMOVE-FROM-WEAK-SET CLOS::ADD-TO-WEAK-SET CLOS::LIST-WEAK-SET)
         3  WEAK-LIST-LIST  (CLOS::REMOVE-FROM-WEAK-SET CLOS::ADD-TO-WEAK-SET CLOS::LIST-WEAK-SET)
         3  WRITE-CHAR-SEQUENCE  (SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER SYSTEM::FILL-STREAM-FLUSH-BUFFER)
         3  MAKE-PIPE-INPUT-STREAM  (SHORT-SITE-NAME LONG-SITE-NAME RUN-SHELL-COMMAND)
         3  SYSTEM::%REMF  (SYSTEM::%SET-DOCUMENTATION CLOS::ENSURE-GENERIC-FUNCTION-USING-CLASS-<T> CLOS::ENSURE-CLASS-USING-CLASS-<T>)
         2  SYSTEM::FUNCTION-BLOCK-NAME  (SYSTEM::MAKE-MACRO-EXPANSION CLOS::ANALYZE-METHOD-DESCRIPTION)
         2  NSTRING-INVERTCASE  (SYSTEM::COMPLETION-4 SYSTEM::COMPLETION)
         2  SYSTEM::READ-FORM  (SYSTEM::STEP-HOOK-FN SYSTEM::DEBUG-RETURN)
         2  SYSTEM::THE-FRAME  (SYSTEM::FRAME-LIMIT-UP SYSTEM::FRAME-LIMIT-DOWN)
         2  SYSTEM::SAME-ENV-AS  (SYSTEM::STEP-HOOK-FN-1 SYSTEM::BREAK-LOOP-2)
         2  SYSTEM::TRAP-EVAL-FRAME  (SYSTEM::DEBUG-TRAP-OFF SYSTEM::DEBUG-TRAP-ON)
         2  GC  (SYSTEM::%SPACE2 SYSTEM::%SPACE1)
         2  READTABLE-CASE  (SYSTEM::COMPLETION SYSTEM::COMPLETION)
         2  LIST-LENGTH-DOTTED  (SYSTEM::DESTRUCTURING-ERROR SYSTEM::MACRO-CALL-ERROR)
         2  LIST-LENGTH-PROPER  (SYSTEM::LIST-TO-HT SYSTEM::LIST-TO-HT)
         2  SYSTEM::LIST-LENGTH-IN-BOUNDS-P  (SYSTEM::FINALIZE-COUTPUT-FILE-9 SYSTEM::CORRECTABLE-ERROR)
         2  SYSTEM::VERSION  (COMPILE-FILE SYSTEM::OPEN-FOR-LOAD-CHECK-COMPILED-FILE)
         2  SYSTEM::DEFAULT-TIME-ZONE  (DECODE-UNIVERSAL-TIME ENCODE-UNIVERSAL-TIME)
         2  SYSTEM::LIB-DIRECTORY  (REQUIRE SYSTEM::CLISP-DATA-FILE)
         2  SYSTEM::GC-STATISTICS  (SYSTEM::%SPACE2 SYSTEM::%SPACE1)
         2  SYSTEM::CLOSURE-CONST  (SYSTEM::%LOCAL-GET SYSTEM::PASS3)
         2  SYSTEM::SET-CLOSURE-CONST  (SYSTEM::PASS3 SYSTEM::%LOCAL-SET)
         2  SYSTEM::CLOSURE-SET-DOCUMENTATION  (COMPILE CLOS::SET-FUNCTION-DOCUMENTATION)
         2  SET-FUNCALLABLE-INSTANCE-FUNCTION  (CLOS::FINALIZE-FAST-GF CLOS::INSTALL-DISPATCH)
         2  SYSTEM::GENERIC-FUNCTION-EFFECTIVE-METHOD-FUNCTION  (CLOS::%CALL-NEXT-METHOD CLOS::%CALL-NEXT-METHOD)
         2  SYSTEM::MAKE-SYMBOL-MACRO  (SYSTEM::%EXPAND-FORM SYSTEM::C-SYMBOL-MACROLET)
         2  SYSTEM::|(SETF WEAK-LIST-LIST)|  (CLOS::REMOVE-FROM-WEAK-SET CLOS::ADD-TO-WEAK-SET)
         2  SYNONYM-STREAM-SYMBOL  (SYSTEM::STREAM-OUTPUT-ELEMENT-TYPE SYSTEM::STREAM-INPUT-ELEMENT-TYPE)
         2  TWO-WAY-STREAM-INPUT-STREAM  (SYSTEM::STREAM-INPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  TWO-WAY-STREAM-OUTPUT-STREAM  (SYSTEM::STREAM-OUTPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  ECHO-STREAM-INPUT-STREAM  (SYSTEM::STREAM-INPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  ECHO-STREAM-OUTPUT-STREAM  (SYSTEM::STREAM-OUTPUT-ELEMENT-TYPE DRIBBLE-STREAM)
         2  SYSTEM::TERMINAL-RAW  (SYSTEM::EXEC-WITH-KEYBOARD SYSTEM::EXEC-WITH-KEYBOARD)
         2  MAKE-PIPE-OUTPUT-STREAM  (SYSTEM::MAKE-PRINTER-STREAM RUN-SHELL-COMMAND)
         2  INTERACTIVE-STREAM-P  (CERROR SYSTEM::BREAK-LOOP)
         2  SYSTEM::RANDOM-POSFIXNUM  (CLOS::SHARED-INITIALIZE-<STANDARD-STABLEHASH> CLOS::MAKE-STRUCTURE-STABLEHASH)
         1  SYSTEM::CHECK-SYMBOL  (CLOS::SHARED-INITIALIZE-<POTENTIAL-CLASS>)
         1  SYSTEM::EVAL-AT  (SYSTEM::STEP-HOOK-FN)
         1  SYSTEM::EVAL-FRAME-P  (SYSTEM::COMMANDS)
         1  SYSTEM::DRIVER-FRAME-P  (SYSTEM::FRAME-LIMIT-UP)
         1  SYSTEM::REDO-EVAL-FRAME  (SYSTEM::DEBUG-REDO)
         1  SYSTEM::RETURN-FROM-EVAL-FRAME  (SYSTEM::DEBUG-RETURN)
         1  SHOW-STACK  (SYSTEM::PRINT-BACKTRACE)
         1  SYSTEM::%ROOM  (ROOM)
         1  MAKE-ENCODING  (SYSTEM::GET-CHARSET-RANGE)
         1  SYSTEM::CHARSET-TYPEP  (TYPEP)
         1  SYSTEM::CHARSET-RANGE  (SYSTEM::GET-CHARSET-RANGE)
         1  SYSTEM::FOREIGN-ENCODING  (FFI::EXEC-WITH-FOREIGN-STRING)
         1  CONVERT-STRING-FROM-BYTES  (OPEN-HTTP)
         1  SYSTEM::CONSES-P  (SYSTEM::DS-TYPEP)
         1  SYSTEM::CURRENT-LANGUAGE  (LOCALIZED)
         1  SYSTEM::DELTA4  (SYSTEM::%TIME)
         1  PACKAGE-LOCK  (SYSTEM::F-SIDE-EFFECT)
         1  PACKAGE-CASE-SENSITIVE-P  (SYSTEM::COMPLETION)
         1  SYSTEM::SYMBOL-VALUE-LOCK  (SYSTEM::SET-CHECK-LOCK)
         1  SYSTEM::CHECK-PACKAGE-LOCK  (SYSTEM::CHECK-REDEFINITION)
         1  DELETE-PACKAGE  (INSPECT)
         1  SYSTEM::PACKAGE-ITERATOR  (SYSTEM::PACKAGE-ITERATOR-FUNCTION)
         1  SYSTEM::PACKAGE-ITERATE  (SYSTEM::PACKAGE-ITERATOR-FUNCTION-1)
         1  LOGICAL-PATHNAME  (SYSTEM::VALID-LOGICAL-PATHNAME-STRING-P-3)
         1  TRANSLATE-LOGICAL-PATHNAME  (SYSTEM::OPEN-FOR-LOAD)
         1  SYSTEM::MAKE-LOGICAL-PATHNAME  (SYSTEM::SET-LOGICAL-PATHNAME-TRANSLATIONS)
         1  USER-HOMEDIR-PATHNAME  (EDITOR-TEMPFILE)
         1  ABSOLUTE-PATHNAME  (SYSTEM::XSTRING)
         1  SYSTEM::SET-LIB-DIRECTORY  (SYSTEM::CLISP-DATA-FILE-1)
         1  SYSTEM::NOTE-NEW-STRUCTURE-CLASS  (CLOS::SHARED-INITIALIZE-<STRUCTURE-CLASS>)
         1  SYSTEM::NOTE-NEW-STANDARD-CLASS  (CLOS::FINALIZE-INSTANCE-SEMI-STANDARD-CLASS)
         1  SYSTEM::LIST-STATISTICS  (SYSTEM::%SPACE)
         1  SYSTEM::HEAP-STATISTICS-STATISTICS  (SYSTEM::%SPACE)
         1  SYSTEM::GC-STATISTICS-STATISTICS  (SYSTEM::%SPACE)
         1  SYSTEM::|(SETF CLOSURE-NAME)|  (SYSTEM::MAKE-PRELIMINARY)
         1  SYSTEM::CONSTANT-INITFUNCTION-P  (SYSTEM::DS-ARG-DEFAULT)
         1  SYSTEM::CLOSURE-DOCUMENTATION  (CLOS::FUNCTION-DOCUMENTATION)
         1  SYSTEM::CLOSURE-LAMBDA-LIST  (ARGLIST)
         1  SYSTEM::GLOBAL-SYMBOL-MACRO-DEFINITION  (SYSTEM::VENV-ASSOC)
         1  SYSTEM::FUNCTION-MACRO-EXPANDER  (SYSTEM::FENV-SEARCH)
         1  FINALIZE  (MAKE-XTERM-IO-STREAM)
         1  CLOS::ALLOCATE-METAOBJECT-INSTANCE  (CLOS::COPY-STANDARD-CLASS)
         1  CLOS::%CHANGE-CLASS  (CLOS::DO-CHANGE-CLASS)
         1  MAKE-WEAK-LIST  (CLOS::ADD-TO-WEAK-SET)
         1  WRITE-BYTE-SEQUENCE  (SYSTEM::HTTP-ERROR)
         1  BROADCAST-STREAM-STREAMS  (DRIBBLE-STREAM)
         1  SYSTEM::MAKE-KEYBOARD-STREAM  (SYSTEM::EXEC-WITH-KEYBOARD)
         1  MAKE-PIPE-IO-STREAM  (RUN-SHELL-COMMAND)
         1  SOCKET-ACCEPT  (SYSTEM::HTTP-COMMAND)
         1  SOCKET-CONNECT  (EXT::OPEN-HTTP-3)
         1  SYSTEM::SET-STREAM-EXTERNAL-FORMAT  (SYSTEM::SET-OUTPUT-STREAM-FASL)
         1  SYSTEM::STREAM-FASL-P  (SYSTEM::SET-OUTPUT-STREAM-FASL)
         1  FFI:VALIDP  (REGEXP::%MATCH)
         1  FFI::CALL-WITH-FOREIGN-STRING  (FFI::EXEC-WITH-FOREIGN-STRING)


)

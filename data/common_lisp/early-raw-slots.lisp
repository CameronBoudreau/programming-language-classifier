;;;; This software is part of the SBCL system. See the README file for
;;;; more information.
;;;;
;;;; This software is derived from the CMU CL system, which was
;;;; written at Carnegie Mellon University and released into the
;;;; public domain. The software is in the public domain and is
;;;; provided with absolutely no warranty. See the COPYING and CREDITS
;;;; files for more information.

(in-package "SB!KERNEL")

;;; STRUCTURE-OBJECT supports placement of raw bits within the object
;;; to allow representation of native word and float-point types directly.

;;; Historically the implementation was optimized for GC by placing all
;;; such slots at the end of the instance, and scavenging only up to last
;;; non-raw slot. This imposed significant overhead for access from Lisp,
;;; because "is-a" inheritance was obliged to rearrange raw slots
;;; to comply with the GC requirement, thus forcing ancestor structure
;;; accessors to compensate for physical structure length in all cases.
;;; Assuming that it is more important to simplify Lisp access than
;;; to simplify GC, we use a more flexible strategy that permits
;;; descendant structures to place new slots anywhere without changing
;;; slot placement established in ancestor structures.
;;; The trade-off is that GC (and a few other things - structure dumping,
;;; EQUALP checking, to name a few) have to be able to determine for each
;;; slot whether it is a Lisp descriptor or just bits. This is done
;;; with the LAYOUT-BITMAP of an object's layout.
;;; The bitmap stores a '1' for each bit representing a raw word,
;;; and could be a BIGNUM given a spectacularly huge structure.

;;; Also note that there are possibly some alignment concerns which must
;;; be accounted for when DEFSTRUCT lays out slots,
;;; by injecting padding words appropriately.
;;; For example COMPLEX-DOUBLE-FLOAT *should* be aligned to twice the
;;; alignment of a DOUBLE-FLOAT. It is not, as things stand,
;;; but this is considered a minor bug.

;; To utilize a word-sized slot in a defstruct without having to resort to
;; writing (myslot :type (unsigned-byte #.sb!vm:n-word-bits)), or even
;; worse (:type #+sb-xc-host <sometype> #-sb-xc-host <othertype>),
;; these abstractions are provided as soon as the raw slots defs are.
;; 'signed-word' is here for companionship - slots of that type are not raw.
(def!type sb!vm:word () `(unsigned-byte ,sb!vm:n-word-bits))
(def!type sb!vm:signed-word () `(signed-byte ,sb!vm:n-word-bits))

;; information about how a slot of a given DSD-RAW-TYPE is to be accessed
(defstruct (raw-slot-data
            (:copier nil)
            (:predicate nil))
  ;; the type specifier, which must specify a numeric type.
  (raw-type (missing-arg) :type symbol :read-only t)
  ;; What operator is used to access a slot of this type?
  (accessor-name (missing-arg) :type symbol :read-only t)
  (init-vop (missing-arg) :type symbol :read-only t)
  ;; How many words are each value of this type?
  (n-words (missing-arg) :type (and index (integer 1)) :read-only t)
  ;; Necessary alignment in units of words.  Note that instances
  ;; themselves are aligned by exactly two words, so specifying more
  ;; than two words here would not work.
  (alignment 1 :type (integer 1 2) :read-only t)
  (comparer (missing-arg) :type function :read-only t))

#!-sb-fluid (declaim (freeze-type raw-slot-data))

;; Simulate DEFINE-LOAD-TIME-GLOBAL - always bound in the image
;; but not eval'd in the compiler.
(defglobal *raw-slot-data* nil)
;; By making this a cold-init function, it is possible to use raw slots
;; in cold toplevel forms.
(defun !raw-slot-data-init ()
  (macrolet ((make-comparer (accessor-name)
               #+sb-xc-host
               `(lambda (x y)
                  (declare (ignore x y))
                  (error "~S comparator called" ',accessor-name))
               #-sb-xc-host
               ;; Not a symbol, because there aren't any so-named functions.
               `(named-lambda ,(string (symbolicate accessor-name "="))
                    (index x y)
                  (declare (optimize speed (safety 0)))
                  (= (,accessor-name x index)
                     (,accessor-name y index)))))
    (let ((double-float-alignment
            ;; white list of architectures that can load unaligned doubles:
            #!+(or x86 x86-64 ppc arm64) 1
            ;; at least sparc, mips and alpha can't:
            #!-(or x86 x86-64 ppc arm64) 2))
     (setq *raw-slot-data*
      (vector
       (make-raw-slot-data :raw-type 'sb!vm:word
                           :accessor-name '%raw-instance-ref/word
                           :init-vop 'sb!vm::raw-instance-init/word
                           :n-words 1
                           :comparer (make-comparer %raw-instance-ref/word))
       (make-raw-slot-data :raw-type 'single-float
                           :accessor-name '%raw-instance-ref/single
                           :init-vop 'sb!vm::raw-instance-init/single
                           ;; KLUDGE: On 64 bit architectures, we
                           ;; could pack two SINGLE-FLOATs into the
                           ;; same word if raw slots were indexed
                           ;; using bytes instead of words.  However,
                           ;; I don't personally find optimizing
                           ;; SINGLE-FLOAT memory usage worthwile
                           ;; enough.  And the other datatype that
                           ;; would really benefit is (UNSIGNED-BYTE
                           ;; 32), but that is a subtype of FIXNUM, so
                           ;; we store it unraw anyway.  :-( -- DFL
                           :n-words 1
                           :comparer (make-comparer %raw-instance-ref/single))
       (make-raw-slot-data :raw-type 'double-float
                           :accessor-name '%raw-instance-ref/double
                           :init-vop 'sb!vm::raw-instance-init/double
                           :alignment double-float-alignment
                           :n-words (/ 8 sb!vm:n-word-bytes)
                           :comparer (make-comparer %raw-instance-ref/double))
       (make-raw-slot-data :raw-type 'complex-single-float
                           :accessor-name '%raw-instance-ref/complex-single
                           :init-vop 'sb!vm::raw-instance-init/complex-single
                           :n-words (/ 8 sb!vm:n-word-bytes)
                           :comparer (make-comparer %raw-instance-ref/complex-single))
       (make-raw-slot-data :raw-type 'complex-double-float
                           :accessor-name '%raw-instance-ref/complex-double
                           :init-vop 'sb!vm::raw-instance-init/complex-double
                           :alignment double-float-alignment
                           :n-words (/ 16 sb!vm:n-word-bytes)
                           :comparer (make-comparer %raw-instance-ref/complex-double))
       #!+long-float
       (make-raw-slot-data :raw-type long-float
                           :accessor-name '%raw-instance-ref/long
                           :init-vop 'sb!vm::raw-instance-init/long
                           :n-words #!+x86 3 #!+sparc 4
                           :comparer (make-comparer %raw-instance-ref/long))
       #!+long-float
       (make-raw-slot-data :raw-type complex-long-float
                           :accessor-name '%raw-instance-ref/complex-long
                           :init-vop 'sb!vm::raw-instance-init/complex-long
                           :n-words #!+x86 6 #!+sparc 8
                           :comparer (make-comparer %raw-instance-ref/complex-long)))))))

#+sb-xc-host (!raw-slot-data-init)
#+sb-xc
(declaim (type (simple-vector #.(length *raw-slot-data*)) *raw-slot-data*))

;; DO-INSTANCE-TAGGED-SLOT iterates over the manifest slots of THING
;; that contain tagged objects. (The LAYOUT does not count as a manifest slot).
;; INDEX-VAR is bound to successive slot-indices,
;; and is usually used as the second argument to %INSTANCE-REF.
;; :PAD, if T, includes a final word that may be present at the end of the
;; structure due to alignment requirements.
;; LAYOUT is optional and somewhat unnecessary, but since some uses of
;; this macro already have a layout in hand, it can be supplied.
;; [If the compiler were smarter about doing fewer memory accesses,
;; there would be no need at all for the LAYOUT - if it had already been
;; accessed, it shouldn't be another memory read]
;;
(defmacro do-instance-tagged-slot ((index-var thing &key layout (pad t)) &body body)
  (with-unique-names (instance bitmap limit)
    `(let* ((,instance ,thing)
            (,bitmap (layout-bitmap ,(or layout `(%instance-layout ,instance))))
            (,limit ,(if pad
                         ;; target instances have an odd number of payload words.
                         `(logior (%instance-length ,instance) #-sb-xc-host 1)
                         `(%instance-length ,instance))))
       (do ((,index-var sb!vm:instance-data-start (1+ ,index-var)))
           ((>= ,index-var ,limit))
         (declare (type index ,index-var))
         (unless (logbitp ,index-var ,bitmap)
           ,@body)))))

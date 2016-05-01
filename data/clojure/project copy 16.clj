(defproject concurrency "1.0.0-SNAPSHOT"
  :description "An array-backed set implementation, analogous to
Clojure's array-map, explored in chapter 3 of 'Clojure Programming' by
Emerick, Carper, and Grand."
  :url "http://github.com/clojurebook/ClojureProgramming"
  :dependencies [[org.clojure/clojure "1.3.0"]]
  :profiles {:1.4 {:dependencies [[org.clojure/clojure "1.4.0-beta6"]]}}
  :run-aliases {:loot com.clojurebook.concurrency.game/-loot-demo
                :battle com.clojurebook.concurrency.game/-battle-demo
                :logged-battle com.clojurebook.concurrency.game-validators/-main})

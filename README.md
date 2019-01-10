# MusicOCR

This is not a finished product, it is just something I work on in my spare time.

The goal is to be able to take a photo of a handwritten piece of music and
produce lilypond code for it. There are probably limits to what can be
achieved here :-) but I'm going to see how far I can get. So far, I've got it
to do a somewhat decent job of identifying the objects to parse (notes, note
heads, accidentals, bar lines, that sort of thing) but I want to do some more 
work training models, and then the next step is to localise items vertically
within the lines (for pitch values).

If you want to give it a try, compile ocr_shell and run it; it needs at least
the path to an image file. ocr_shell is a simple character-based shell that
will bring up visualisations of the state of parsing. A few use cases require
a trained model, which you can make for yourself using train_knn (the name is
 misleading, it trains svm and dtrees models too). I have found dtree models to work best in practice.

In order to compile, you need opencv including contrib directories (for the
tesseract interaction). You'll see that I have hardcoded the directories for
those in CMakeLists.txt, you'll need to adapt that for your computer. You'll
also need the tesseract headers where cmake can find them.


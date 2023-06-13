from rwkv_tokenizer import get_tokenizer

tokenizer, tokenizer_encode = get_tokenizer("world")

print('Unit test...')

# Test string taken from https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_tokenizer.py
test_string = '''
UTF-8 decoder capability and stress test
----------------------------------------

Markus Kuhn <http://www.cl.cam.ac.uk/~mgk25/> - 2015-08-28 - CC BY 4.0

This test file can help you examine, how your UTF-8 decoder handles
various types of correct, malformed, or otherwise interesting UTF-8
sequences. This file is not meant to be a conformance test. It does
not prescribe any particular outcome. Therefore, there is no way to
"pass" or "fail" this test file, even though the text does suggest a
preferable decoder behaviour at some places. Its aim is, instead, to
help you think about, and test, the behaviour of your UTF-8 decoder on a
systematic collection of unusual inputs. Experience so far suggests
that most first-time authors of UTF-8 decoders find at least one
serious problem in their decoder using this file.

The test lines below cover boundary conditions, malformed UTF-8
sequences, as well as correctly encoded UTF-8 sequences of Unicode code
points that should never occur in a correct UTF-8 file.

According to ISO 10646-1:2000, sections D.7 and 2.3c, a device
receiving UTF-8 shall interpret a "malformed sequence in the same way
that it interprets a character that is outside the adopted subset" and
"characters that are not within the adopted subset shall be indicated
to the user" by a receiving device. One commonly used approach in
UTF-8 decoders is to replace any malformed UTF-8 sequence by a
replacement character (U+FFFD), which looks a bit like an inverted
question mark, or a similar symbol. It might be a good idea to
visually distinguish a malformed UTF-8 sequence from a correctly
encoded Unicode character that is just not available in the current
font but otherwise fully legal, even though ISO 10646-1 doesn't
mandate this. In any case, just ignoring malformed sequences or
unavailable characters does not conform to ISO 10646, will make
debugging more difficult, and can lead to user confusion.

Please check, whether a malformed UTF-8 sequence is (1) represented at
all, (2) represented by exactly one single replacement character (or
equivalent signal), and (3) the following quotation mark after an
illegal UTF-8 sequence is correctly displayed, i.e. proper
resynchronization takes place immediately after any malformed
sequence. This file says "THE END" in the last line, so if you don't
see that, your decoder crashed somehow before, which should always be
cause for concern.

All lines in this file are exactly 79 characters long (plus the line
feed). In addition, all lines end with "|", except for the two test
lines 2.1.1 and 2.2.1, which contain non-printable ASCII controls
U+0000 and U+007F. If you display this file with a fixed-width font,
these "|" characters should all line up in column 79 (right margin).
This allows you to test quickly, whether your UTF-8 decoder finds the
correct number of characters in every line, that is whether each
malformed sequences is replaced by a single replacement character.

Note that, as an alternative to the notion of malformed sequence used
here, it is also a perfectly acceptable (and in some situations even
preferable) solution to represent each individual byte of a malformed
sequence with a replacement character. If you follow this strategy in
your decoder, then please ignore the "|" column.


Here come the tests:                                                          |
                                                                              |
1  Some correct UTF-8 text                                                    |
                                                                              |
You should see the Greek word 'kosme':       "κόσμε"                          |
                                                                              |
2  Boundary condition test cases                                              |
                                                                              |
2.1  First possible sequence of a certain length                              |
                                                                              |
2.1.1  1 byte  (U-00000000):        "�"                                        
2.1.2  2 bytes (U-00000080):        ""                                       |
2.1.3  3 bytes (U-00000800):        "ࠀ"                                       |
2.1.4  4 bytes (U-00010000):        "𐀀"                                       |
2.1.5  5 bytes (U-00200000):        "�����"                                       |
2.1.6  6 bytes (U-04000000):        "������"                                       |
                                                                              |
2.2  Last possible sequence of a certain length                               |
                                                                              |
2.2.1  1 byte  (U-0000007F):        ""                                        
2.2.2  2 bytes (U-000007FF):        "߿"                                       |
2.2.3  3 bytes (U-0000FFFF):        "￿"                                       |
2.2.4  4 bytes (U-001FFFFF):        "����"                                       |
2.2.5  5 bytes (U-03FFFFFF):        "�����"                                       |
2.2.6  6 bytes (U-7FFFFFFF):        "������"                                       |
                                                                              |
2.3  Other boundary conditions                                                |
                                                                              |
2.3.1  U-0000D7FF = ed 9f bf = "퟿"                                            |
2.3.2  U-0000E000 = ee 80 80 = ""                                            |
2.3.3  U-0000FFFD = ef bf bd = "�"                                            |
2.3.4  U-0010FFFF = f4 8f bf bf = "􏿿"                                         |
2.3.5  U-00110000 = f4 90 80 80 = "����"                                         |
                                                                              |
3  Malformed sequences                                                        |
                                                                              |
3.1  Unexpected continuation bytes                                            |
                                                                              |
Each unexpected continuation byte should be separately signalled as a         |
malformed sequence of its own.                                                |
                                                                              |
3.1.1  First continuation byte 0x80: "�"                                      |
3.1.2  Last  continuation byte 0xbf: "�"                                      |
                                                                              |
3.1.3  2 continuation bytes: "��"                                             |
3.1.4  3 continuation bytes: "���"                                            |
3.1.5  4 continuation bytes: "����"                                           |
3.1.6  5 continuation bytes: "�����"                                          |
3.1.7  6 continuation bytes: "������"                                         |
3.1.8  7 continuation bytes: "�������"                                        |
                                                                              |
3.1.9  Sequence of all 64 possible continuation bytes (0x80-0xbf):            |
                                                                              |
   "����������������                                                          |
    ����������������                                                          |
    ����������������                                                          |
    ����������������"                                                         |
                                                                              |
3.2  Lonely start characters                                                  |
                                                                              |
3.2.1  All 32 first bytes of 2-byte sequences (0xc0-0xdf),                    |
       each followed by a space character:                                    |
                                                                              |
   "� � � � � � � � � � � � � � � �                                           |
    � � � � � � � � � � � � � � � � "                                         |
                                                                              |
3.2.2  All 16 first bytes of 3-byte sequences (0xe0-0xef),                    |
       each followed by a space character:                                    |
                                                                              |
   "� � � � � � � � � � � � � � � � "                                         |
                                                                              |
3.2.3  All 8 first bytes of 4-byte sequences (0xf0-0xf7),                     |
       each followed by a space character:                                    |
                                                                              |
   "� � � � � � � � "                                                         |
                                                                              |
3.2.4  All 4 first bytes of 5-byte sequences (0xf8-0xfb),                     |
       each followed by a space character:                                    |
                                                                              |
   "� � � � "                                                                 |
                                                                              |
3.2.5  All 2 first bytes of 6-byte sequences (0xfc-0xfd),                     |
       each followed by a space character:                                    |
                                                                              |
   "� � "                                                                     |
                                                                              |
3.3  Sequences with last continuation byte missing                            |
                                                                              |
All bytes of an incomplete sequence should be signalled as a single           |
malformed sequence, i.e., you should see only a single replacement            |
character in each of the next 10 tests. (Characters as in section 2)          |
                                                                              |
3.3.1  2-byte sequence with last byte missing (U+0000):     "�"               |
3.3.2  3-byte sequence with last byte missing (U+0000):     "��"               |
3.3.3  4-byte sequence with last byte missing (U+0000):     "���"               |
3.3.4  5-byte sequence with last byte missing (U+0000):     "����"               |
3.3.5  6-byte sequence with last byte missing (U+0000):     "�����"               |
3.3.6  2-byte sequence with last byte missing (U-000007FF): "�"               |
3.3.7  3-byte sequence with last byte missing (U-0000FFFF): "�"               |
3.3.8  4-byte sequence with last byte missing (U-001FFFFF): "���"               |
3.3.9  5-byte sequence with last byte missing (U-03FFFFFF): "����"               |
3.3.10 6-byte sequence with last byte missing (U-7FFFFFFF): "�����"               |
                                                                              |
3.4  Concatenation of incomplete sequences                                    |
                                                                              |
All the 10 sequences of 3.3 concatenated, you should see 10 malformed         |
sequences being signalled:                                                    |
                                                                              |
   "�����������������������������"                                                               |
                                                                              |
3.5  Impossible bytes                                                         |
                                                                              |
The following two bytes cannot appear in a correct UTF-8 string               |
                                                                              |
3.5.1  fe = "�"                                                               |
3.5.2  ff = "�"                                                               |
3.5.3  fe fe ff ff = "����"                                                   |
                                                                              |
4  Overlong sequences                                                         |
                                                                              |
The following sequences are not malformed according to the letter of          |
the Unicode 2.0 standard. However, they are longer then necessary and         |
a correct UTF-8 encoder is not allowed to produce them. A "safe UTF-8         |
decoder" should reject them just like malformed sequences for two             |
reasons: (1) It helps to debug applications if overlong sequences are         |
not treated as valid representations of characters, because this helps        |
to spot problems more quickly. (2) Overlong sequences provide                 |
alternative representations of characters, that could maliciously be          |
used to bypass filters that check only for ASCII characters. For              |
instance, a 2-byte encoded line feed (LF) would not be caught by a            |
line counter that counts only 0x0a bytes, but it would still be               |
processed as a line feed by an unsafe UTF-8 decoder later in the              |
pipeline. From a security point of view, ASCII compatibility of UTF-8         |
sequences means also, that ASCII characters are *only* allowed to be          |
represented by ASCII bytes in the range 0x00-0x7f. To ensure this             |
aspect of ASCII compatibility, use only "safe UTF-8 decoders" that            |
reject overlong UTF-8 sequences for which a shorter encoding exists.          |
                                                                              |
4.1  Examples of an overlong ASCII character                                  |
                                                                              |
With a safe UTF-8 decoder, all of the following five overlong                 |
representations of the ASCII character slash ("/") should be rejected         |
like a malformed UTF-8 sequence, for instance by substituting it with         |
a replacement character. If you see a slash below, you do not have a          |
safe UTF-8 decoder!                                                           |
                                                                              |
4.1.1 U+002F = c0 af             = "��"                                        |
4.1.2 U+002F = e0 80 af          = "���"                                        |
4.1.3 U+002F = f0 80 80 af       = "����"                                        |
4.1.4 U+002F = f8 80 80 80 af    = "�����"                                        |
4.1.5 U+002F = fc 80 80 80 80 af = "������"                                        |
                                                                              |
4.2  Maximum overlong sequences                                               |
                                                                              |
Below you see the highest Unicode value that is still resulting in an         |
overlong sequence if represented with the given number of bytes. This         |
is a boundary test for safe UTF-8 decoders. All five characters should        |
be rejected like malformed UTF-8 sequences.                                   |
                                                                              |
4.2.1  U-0000007F = c1 bf             = "��"                                   |
4.2.2  U-000007FF = e0 9f bf          = "���"                                   |
4.2.3  U-0000FFFF = f0 8f bf bf       = "����"                                   |
4.2.4  U-001FFFFF = f8 87 bf bf bf    = "�����"                                   |
4.2.5  U-03FFFFFF = fc 83 bf bf bf bf = "������"                                   |
                                                                              |
4.3  Overlong representation of the NUL character                             |
                                                                              |
The following five sequences should also be rejected like malformed           |
UTF-8 sequences and should not be treated like the ASCII NUL                  |
character.                                                                    |
                                                                              |
4.3.1  U+0000 = c0 80             = "��"                                       |
4.3.2  U+0000 = e0 80 80          = "���"                                       |
4.3.3  U+0000 = f0 80 80 80       = "����"                                       |
4.3.4  U+0000 = f8 80 80 80 80    = "�����"                                       |
4.3.5  U+0000 = fc 80 80 80 80 80 = "������"                                       |
                                                                              |
5  Illegal code positions                                                     |
                                                                              |
The following UTF-8 sequences should be rejected like malformed               |
sequences, because they never represent valid ISO 10646 characters and        |
a UTF-8 decoder that accepts them might introduce security problems           |
comparable to overlong UTF-8 sequences.                                       |
                                                                              |
5.1 Single UTF-16 surrogates                                                  |
                                                                              |
5.1.1  U+D800 = ed a0 80 = "���"                                                |
5.1.2  U+DB7F = ed ad bf = "���"                                                |
5.1.3  U+DB80 = ed ae 80 = "���"                                                |
5.1.4  U+DBFF = ed af bf = "���"                                                |
5.1.5  U+DC00 = ed b0 80 = "���"                                                |
5.1.6  U+DF80 = ed be 80 = "���"                                                |
5.1.7  U+DFFF = ed bf bf = "���"                                                |
                                                                              |
5.2 Paired UTF-16 surrogates                                                  |
                                                                              |
5.2.1  U+D800 U+DC00 = ed a0 80 ed b0 80 = "������"                               |
5.2.2  U+D800 U+DFFF = ed a0 80 ed bf bf = "������"                               |
5.2.3  U+DB7F U+DC00 = ed ad bf ed b0 80 = "������"                               |
5.2.4  U+DB7F U+DFFF = ed ad bf ed bf bf = "������"                               |
5.2.5  U+DB80 U+DC00 = ed ae 80 ed b0 80 = "������"                               |
5.2.6  U+DB80 U+DFFF = ed ae 80 ed bf bf = "������"                               |
5.2.7  U+DBFF U+DC00 = ed af bf ed b0 80 = "������"                               |
5.2.8  U+DBFF U+DFFF = ed af bf ed bf bf = "������"                               |
                                                                              |
5.3 Noncharacter code positions                                               |
                                                                              |
The following "noncharacters" are "reserved for internal use" by              |
applications, and according to older versions of the Unicode Standard         |
"should never be interchanged". Unicode Corrigendum #9 dropped the            |
latter restriction. Nevertheless, their presence in incoming UTF-8 data       |
can remain a potential security risk, depending on what use is made of        |
these codes subsequently. Examples of such internal use:                      |
                                                                              |
 - Some file APIs with 16-bit characters may use the integer value -1         |
   = U+FFFF to signal an end-of-file (EOF) or error condition.                |
                                                                              |
 - In some UTF-16 receivers, code point U+FFFE might trigger a                |
   byte-swap operation (to convert between UTF-16LE and UTF-16BE).            |
                                                                              |
With such internal use of noncharacters, it may be desirable and safer        |
to block those code points in UTF-8 decoders, as they should never            |
occur legitimately in incoming UTF-8 data, and could trigger unsafe           |
behaviour in subsequent processing.                                           |
                                                                              |
Particularly problematic noncharacters in 16-bit applications:                |
                                                                              |
5.3.1  U+FFFE = ef bf be = "￾"                                                |
5.3.2  U+FFFF = ef bf bf = "￿"                                                |
                                                                              |
Other noncharacters:                                                          |
                                                                              |
5.3.3  U+FDD0 .. U+FDEF = "﷐﷑﷒﷓﷔﷕﷖﷗﷘﷙﷚﷛﷜﷝﷞﷟﷠﷡﷢﷣﷤﷥﷦﷧﷨﷩﷪﷫﷬﷭﷮﷯"|
                                                                              |
5.3.4  U+nFFFE U+nFFFF (for n = 1..10)                                        |
                                                                              |
       "🿾🿿𯿾𯿿𿿾𿿿񏿾񏿿񟿾񟿿񯿾񯿿񿿾񿿿򏿾򏿿                                    |
        򟿾򟿿򯿾򯿿򿿾򿿿󏿾󏿿󟿾󟿿󯿾󯿿󿿾󿿿􏿾􏿿"                                   |
                                                                              |
THE END                                                                       |


UTF-8 encoded sample plain-text file
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

Markus Kuhn [ˈmaʳkʊs kuːn] <http://www.cl.cam.ac.uk/~mgk25/> — 2002-07-25 CC BY


The ASCII compatible UTF-8 encoding used in this plain-text file
is defined in Unicode, ISO 10646-1, and RFC 2279.


Using Unicode/UTF-8, you can write in emails and source code things such as

Mathematics and sciences:

  ∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i),      ⎧⎡⎛┌─────┐⎞⎤⎫
                                            ⎪⎢⎜│a²+b³ ⎟⎥⎪
  ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β),    ⎪⎢⎜│───── ⎟⎥⎪
                                            ⎪⎢⎜⎷ c₈   ⎟⎥⎪
  ℕ ⊆ ℕ₀ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ,                   ⎨⎢⎜       ⎟⎥⎬
                                            ⎪⎢⎜ ∞     ⎟⎥⎪
  ⊥ < a ≠ b ≡ c ≤ d ≪ ⊤ ⇒ (⟦A⟧ ⇔ ⟪B⟫),      ⎪⎢⎜ ⎲     ⎟⎥⎪
                                            ⎪⎢⎜ ⎳aⁱ-bⁱ⎟⎥⎪
  2H₂ + O₂ ⇌ 2H₂O, R = 4.7 kΩ, ⌀ 200 mm     ⎩⎣⎝i=1    ⎠⎦⎭

Linguistics and dictionaries:

  ði ıntəˈnæʃənəl fəˈnɛtık əsoʊsiˈeıʃn
  Y [ˈʏpsilɔn], Yen [jɛn], Yoga [ˈjoːgɑ]

APL:

  ((V⍳V)=⍳⍴V)/V←,V    ⌷←⍳→⍴∆∇⊃‾⍎⍕⌈

Nicer typography in plain text files:

  ╔══════════════════════════════════════════╗
  ║                                          ║
  ║   • ‘single’ and “double” quotes         ║
  ║                                          ║
  ║   • Curly apostrophes: “We’ve been here” ║
  ║                                          ║
  ║   • Latin-1 apostrophe and accents: '´`  ║
  ║                                          ║
  ║   • ‚deutsche‘ „Anführungszeichen“       ║
  ║                                          ║
  ║   • †, ‡, ‰, •, 3–4, —, −5/+5, ™, …      ║
  ║                                          ║
  ║   • ASCII safety test: 1lI|, 0OD, 8B     ║
  ║                      ╭─────────╮         ║
  ║   • the euro symbol: │ 14.95 € │         ║
  ║                      ╰─────────╯         ║
  ╚══════════════════════════════════════════╝

Combining characters:

  STARGΛ̊TE SG-1, a = v̇ = r̈, a⃑ ⊥ b⃑

Greek (in Polytonic):

  The Greek anthem:

  Σὲ γνωρίζω ἀπὸ τὴν κόψη
  τοῦ σπαθιοῦ τὴν τρομερή,
  σὲ γνωρίζω ἀπὸ τὴν ὄψη
  ποὺ μὲ βία μετράει τὴ γῆ.

  ᾿Απ᾿ τὰ κόκκαλα βγαλμένη
  τῶν ῾Ελλήνων τὰ ἱερά
  καὶ σὰν πρῶτα ἀνδρειωμένη
  χαῖρε, ὦ χαῖρε, ᾿Ελευθεριά!

  From a speech of Demosthenes in the 4th century BC:

  Οὐχὶ ταὐτὰ παρίσταταί μοι γιγνώσκειν, ὦ ἄνδρες ᾿Αθηναῖοι,
  ὅταν τ᾿ εἰς τὰ πράγματα ἀποβλέψω καὶ ὅταν πρὸς τοὺς
  λόγους οὓς ἀκούω· τοὺς μὲν γὰρ λόγους περὶ τοῦ
  τιμωρήσασθαι Φίλιππον ὁρῶ γιγνομένους, τὰ δὲ πράγματ᾿
  εἰς τοῦτο προήκοντα,  ὥσθ᾿ ὅπως μὴ πεισόμεθ᾿ αὐτοὶ
  πρότερον κακῶς σκέψασθαι δέον. οὐδέν οὖν ἄλλο μοι δοκοῦσιν
  οἱ τὰ τοιαῦτα λέγοντες ἢ τὴν ὑπόθεσιν, περὶ ἧς βουλεύεσθαι,
  οὐχὶ τὴν οὖσαν παριστάντες ὑμῖν ἁμαρτάνειν. ἐγὼ δέ, ὅτι μέν
  ποτ᾿ ἐξῆν τῇ πόλει καὶ τὰ αὑτῆς ἔχειν ἀσφαλῶς καὶ Φίλιππον
  τιμωρήσασθαι, καὶ μάλ᾿ ἀκριβῶς οἶδα· ἐπ᾿ ἐμοῦ γάρ, οὐ πάλαι
  γέγονεν ταῦτ᾿ ἀμφότερα· νῦν μέντοι πέπεισμαι τοῦθ᾿ ἱκανὸν
  προλαβεῖν ἡμῖν εἶναι τὴν πρώτην, ὅπως τοὺς συμμάχους
  σώσομεν. ἐὰν γὰρ τοῦτο βεβαίως ὑπάρξῃ, τότε καὶ περὶ τοῦ
  τίνα τιμωρήσεταί τις καὶ ὃν τρόπον ἐξέσται σκοπεῖν· πρὶν δὲ
  τὴν ἀρχὴν ὀρθῶς ὑποθέσθαι, μάταιον ἡγοῦμαι περὶ τῆς
  τελευτῆς ὁντινοῦν ποιεῖσθαι λόγον.

  Δημοσθένους, Γ´ ᾿Ολυνθιακὸς

Georgian:

  From a Unicode conference invitation:

  გთხოვთ ახლავე გაიაროთ რეგისტრაცია Unicode-ის მეათე საერთაშორისო
  კონფერენციაზე დასასწრებად, რომელიც გაიმართება 10-12 მარტს,
  ქ. მაინცში, გერმანიაში. კონფერენცია შეჰკრებს ერთად მსოფლიოს
  ექსპერტებს ისეთ დარგებში როგორიცაა ინტერნეტი და Unicode-ი,
  ინტერნაციონალიზაცია და ლოკალიზაცია, Unicode-ის გამოყენება
  ოპერაციულ სისტემებსა, და გამოყენებით პროგრამებში, შრიფტებში,
  ტექსტების დამუშავებასა და მრავალენოვან კომპიუტერულ სისტემებში.

Russian:

  From a Unicode conference invitation:

  Зарегистрируйтесь сейчас на Десятую Международную Конференцию по
  Unicode, которая состоится 10-12 марта 1997 года в Майнце в Германии.
  Конференция соберет широкий круг экспертов по  вопросам глобального
  Интернета и Unicode, локализации и интернационализации, воплощению и
  применению Unicode в различных операционных системах и программных
  приложениях, шрифтах, верстке и многоязычных компьютерных системах.

Thai (UCS Level 2):

  Excerpt from a poetry on The Romance of The Three Kingdoms (a Chinese
  classic 'San Gua'):

  [----------------------------|------------------------]
    ๏ แผ่นดินฮั่นเสื่อมโทรมแสนสังเวช  พระปกเกศกองบู๊กู้ขึ้นใหม่
  สิบสองกษัตริย์ก่อนหน้าแลถัดไป       สององค์ไซร้โง่เขลาเบาปัญญา
    ทรงนับถือขันทีเป็นที่พึ่ง           บ้านเมืองจึงวิปริตเป็นนักหนา
  โฮจิ๋นเรียกทัพทั่วหัวเมืองมา         หมายจะฆ่ามดชั่วตัวสำคัญ
    เหมือนขับไสไล่เสือจากเคหา      รับหมาป่าเข้ามาเลยอาสัญ
  ฝ่ายอ้องอุ้นยุแยกให้แตกกัน          ใช้สาวนั้นเป็นชนวนชื่นชวนใจ
    พลันลิฉุยกุยกีกลับก่อเหตุ          ช่างอาเพศจริงหนาฟ้าร้องไห้
  ต้องรบราฆ่าฟันจนบรรลัย           ฤๅหาใครค้ำชูกู้บรรลังก์ ฯ

  (The above is a two-column text. If combining characters are handled
  correctly, the lines of the second column should be aligned with the
  | character above.)

Ethiopian:

  Proverbs in the Amharic language:

  ሰማይ አይታረስ ንጉሥ አይከሰስ።
  ብላ ካለኝ እንደአባቴ በቆመጠኝ።
  ጌጥ ያለቤቱ ቁምጥና ነው።
  ደሀ በሕልሙ ቅቤ ባይጠጣ ንጣት በገደለው።
  የአፍ ወለምታ በቅቤ አይታሽም።
  አይጥ በበላ ዳዋ ተመታ።
  ሲተረጉሙ ይደረግሙ።
  ቀስ በቀስ፥ ዕንቁላል በእግሩ ይሄዳል።
  ድር ቢያብር አንበሳ ያስር።
  ሰው እንደቤቱ እንጅ እንደ ጉረቤቱ አይተዳደርም።
  እግዜር የከፈተውን ጉሮሮ ሳይዘጋው አይድርም።
  የጎረቤት ሌባ፥ ቢያዩት ይስቅ ባያዩት ያጠልቅ።
  ሥራ ከመፍታት ልጄን ላፋታት።
  ዓባይ ማደሪያ የለው፥ ግንድ ይዞ ይዞራል።
  የእስላም አገሩ መካ የአሞራ አገሩ ዋርካ።
  ተንጋሎ ቢተፉ ተመልሶ ባፉ።
  ወዳጅህ ማር ቢሆን ጨርስህ አትላሰው።
  እግርህን በፍራሽህ ልክ ዘርጋ።

Runes:

  ᚻᛖ ᚳᚹᚫᚦ ᚦᚫᛏ ᚻᛖ ᛒᚢᛞᛖ ᚩᚾ ᚦᚫᛗ ᛚᚪᚾᛞᛖ ᚾᚩᚱᚦᚹᛖᚪᚱᛞᚢᛗ ᚹᛁᚦ ᚦᚪ ᚹᛖᛥᚫ

  (Old English, which transcribed into Latin reads 'He cwaeth that he
  bude thaem lande northweardum with tha Westsae.' and means 'He said
  that he lived in the northern land near the Western Sea.')

Braille:

  ⡌⠁⠧⠑ ⠼⠁⠒  ⡍⠜⠇⠑⠹⠰⠎ ⡣⠕⠌

  ⡍⠜⠇⠑⠹ ⠺⠁⠎ ⠙⠑⠁⠙⠒ ⠞⠕ ⠃⠑⠛⠔ ⠺⠊⠹⠲ ⡹⠻⠑ ⠊⠎ ⠝⠕ ⠙⠳⠃⠞
  ⠱⠁⠞⠑⠧⠻ ⠁⠃⠳⠞ ⠹⠁⠞⠲ ⡹⠑ ⠗⠑⠛⠊⠌⠻ ⠕⠋ ⠙⠊⠎ ⠃⠥⠗⠊⠁⠇ ⠺⠁⠎
  ⠎⠊⠛⠝⠫ ⠃⠹ ⠹⠑ ⠊⠇⠻⠛⠹⠍⠁⠝⠂ ⠹⠑ ⠊⠇⠻⠅⠂ ⠹⠑ ⠥⠝⠙⠻⠞⠁⠅⠻⠂
  ⠁⠝⠙ ⠹⠑ ⠡⠊⠑⠋ ⠍⠳⠗⠝⠻⠲ ⡎⠊⠗⠕⠕⠛⠑ ⠎⠊⠛⠝⠫ ⠊⠞⠲ ⡁⠝⠙
  ⡎⠊⠗⠕⠕⠛⠑⠰⠎ ⠝⠁⠍⠑ ⠺⠁⠎ ⠛⠕⠕⠙ ⠥⠏⠕⠝ ⠰⡡⠁⠝⠛⠑⠂ ⠋⠕⠗ ⠁⠝⠹⠹⠔⠛ ⠙⠑
  ⠡⠕⠎⠑ ⠞⠕ ⠏⠥⠞ ⠙⠊⠎ ⠙⠁⠝⠙ ⠞⠕⠲

  ⡕⠇⠙ ⡍⠜⠇⠑⠹ ⠺⠁⠎ ⠁⠎ ⠙⠑⠁⠙ ⠁⠎ ⠁ ⠙⠕⠕⠗⠤⠝⠁⠊⠇⠲

  ⡍⠔⠙⠖ ⡊ ⠙⠕⠝⠰⠞ ⠍⠑⠁⠝ ⠞⠕ ⠎⠁⠹ ⠹⠁⠞ ⡊ ⠅⠝⠪⠂ ⠕⠋ ⠍⠹
  ⠪⠝ ⠅⠝⠪⠇⠫⠛⠑⠂ ⠱⠁⠞ ⠹⠻⠑ ⠊⠎ ⠏⠜⠞⠊⠊⠥⠇⠜⠇⠹ ⠙⠑⠁⠙ ⠁⠃⠳⠞
  ⠁ ⠙⠕⠕⠗⠤⠝⠁⠊⠇⠲ ⡊ ⠍⠊⠣⠞ ⠙⠁⠧⠑ ⠃⠑⠲ ⠔⠊⠇⠔⠫⠂ ⠍⠹⠎⠑⠇⠋⠂ ⠞⠕
  ⠗⠑⠛⠜⠙ ⠁ ⠊⠕⠋⠋⠔⠤⠝⠁⠊⠇ ⠁⠎ ⠹⠑ ⠙⠑⠁⠙⠑⠌ ⠏⠊⠑⠊⠑ ⠕⠋ ⠊⠗⠕⠝⠍⠕⠝⠛⠻⠹
  ⠔ ⠹⠑ ⠞⠗⠁⠙⠑⠲ ⡃⠥⠞ ⠹⠑ ⠺⠊⠎⠙⠕⠍ ⠕⠋ ⠳⠗ ⠁⠝⠊⠑⠌⠕⠗⠎
  ⠊⠎ ⠔ ⠹⠑ ⠎⠊⠍⠊⠇⠑⠆ ⠁⠝⠙ ⠍⠹ ⠥⠝⠙⠁⠇⠇⠪⠫ ⠙⠁⠝⠙⠎
  ⠩⠁⠇⠇ ⠝⠕⠞ ⠙⠊⠌⠥⠗⠃ ⠊⠞⠂ ⠕⠗ ⠹⠑ ⡊⠳⠝⠞⠗⠹⠰⠎ ⠙⠕⠝⠑ ⠋⠕⠗⠲ ⡹⠳
  ⠺⠊⠇⠇ ⠹⠻⠑⠋⠕⠗⠑ ⠏⠻⠍⠊⠞ ⠍⠑ ⠞⠕ ⠗⠑⠏⠑⠁⠞⠂ ⠑⠍⠏⠙⠁⠞⠊⠊⠁⠇⠇⠹⠂ ⠹⠁⠞
  ⡍⠜⠇⠑⠹ ⠺⠁⠎ ⠁⠎ ⠙⠑⠁⠙ ⠁⠎ ⠁ ⠙⠕⠕⠗⠤⠝⠁⠊⠇⠲

  (The first couple of paragraphs of "A Christmas Carol" by Dickens)

Compact font selection example text:

  ABCDEFGHIJKLMNOPQRSTUVWXYZ /0123456789
  abcdefghijklmnopqrstuvwxyz £©µÀÆÖÞßéöÿ
  –—‘“”„†•…‰™œŠŸž€ ΑΒΓΔΩαβγδω АБВГДабвгд
  ∀∂∈ℝ∧∪≡∞ ↑↗↨↻⇣ ┐┼╔╘░►☺♀ ﬁ�⑀₂ἠḂӥẄɐː⍎אԱა

Greetings in various languages:

  Hello world, Καλημέρα κόσμε, コンニチハ

Box drawing alignment tests:                                          █
                                                                      ▉
  ╔══╦══╗  ┌──┬──┐  ╭──┬──╮  ╭──┬──╮  ┏━━┳━━┓  ┎┒┏┑   ╷  ╻ ┏┯┓ ┌┰┐    ▊ ╱╲╱╲╳╳╳
  ║┌─╨─┐║  │╔═╧═╗│  │╒═╪═╕│  │╓─╁─╖│  ┃┌─╂─┐┃  ┗╃╄┙  ╶┼╴╺╋╸┠┼┨ ┝╋┥    ▋ ╲╱╲╱╳╳╳
  ║│╲ ╱│║  │║   ║│  ││ │ ││  │║ ┃ ║│  ┃│ ╿ │┃  ┍╅╆┓   ╵  ╹ ┗┷┛ └┸┘    ▌ ╱╲╱╲╳╳╳
  ╠╡ ╳ ╞╣  ├╢   ╟┤  ├┼─┼─┼┤  ├╫─╂─╫┤  ┣┿╾┼╼┿┫  ┕┛┖┚     ┌┄┄┐ ╎ ┏┅┅┓ ┋ ▍ ╲╱╲╱╳╳╳
  ║│╱ ╲│║  │║   ║│  ││ │ ││  │║ ┃ ║│  ┃│ ╽ │┃  ░░▒▒▓▓██ ┊  ┆ ╎ ╏  ┇ ┋ ▎
  ║└─╥─┘║  │╚═╤═╝│  │╘═╪═╛│  │╙─╀─╜│  ┃└─╂─┘┃  ░░▒▒▓▓██ ┊  ┆ ╎ ╏  ┇ ┋ ▏
  ╚══╩══╝  └──┴──┘  ╰──┴──╯  ╰──┴──╯  ┗━━┻━━┛  ▗▄▖▛▀▜   └╌╌┘ ╎ ┗╍╍┛ ┋  ▁▂▃▄▅▆▇█
                                               ▝▀▘▙▄▟

Sanskrit: ﻿काचं शक्नोम्यत्तुम् । नोपहिनस्ति माम् ॥
Sanskrit (standard transcription): kācaṃ śaknomyattum; nopahinasti mām.
Classical Greek: ὕαλον ϕαγεῖν δύναμαι· τοῦτο οὔ με βλάπτει.
Greek (monotonic): Μπορώ να φάω σπασμένα γυαλιά χωρίς να πάθω τίποτα.
Greek (polytonic): Μπορῶ νὰ φάω σπασμένα γυαλιὰ χωρὶς νὰ πάθω τίποτα.
Etruscan: (NEEDED)
Latin: Vitrum edere possum; mihi non nocet.
Old French: Je puis mangier del voirre. Ne me nuit.
French: Je peux manger du verre, ça ne me fait pas mal.
Provençal / Occitan: Pòdi manjar de veire, me nafrariá pas.
Québécois: J'peux manger d'la vitre, ça m'fa pas mal.
Walloon: Dji pou magnî do vêre, çoula m' freut nén må.
Champenois: (NEEDED)
Lorrain: (NEEDED)
Picard: Ch'peux mingi du verre, cha m'foé mie n'ma.
Corsican/Corsu: (NEEDED)
Jèrriais: (NEEDED)
Kreyòl Ayisyen (Haitï): Mwen kap manje vè, li pa blese'm.
Basque: Kristala jan dezaket, ez dit minik ematen.
Catalan / Català: Puc menjar vidre, que no em fa mal.
Spanish: Puedo comer vidrio, no me hace daño.
Aragonés: Puedo minchar beire, no me'n fa mal .
Aranés: (NEEDED)
Mallorquín: (NEEDED)
Galician: Eu podo xantar cristais e non cortarme.
European Portuguese: Posso comer vidro, não me faz mal.
Brazilian Portuguese (8): Posso comer vidro, não me machuca.
Caboverdiano/Kabuverdianu (Cape Verde): M' podê cumê vidru, ca ta maguâ-m'.
Papiamentu: Ami por kome glas anto e no ta hasimi daño.
Italian: Posso mangiare il vetro e non mi fa male.
Milanese: Sôn bôn de magnà el véder, el me fa minga mal.
Roman: Me posso magna' er vetro, e nun me fa male.
Napoletano: M' pozz magna' o'vetr, e nun m' fa mal.
Venetian: Mi posso magnare el vetro, no'l me fa mae.
Zeneise (Genovese): Pòsso mangiâ o veddro e o no me fà mâ.
Sicilian: Puotsu mangiari u vitru, nun mi fa mali.
Campinadese (Sardinia): (NEEDED)
Lugudorese (Sardinia): (NEEDED)
Romansch (Grischun): Jau sai mangiar vaider, senza che quai fa donn a mai.
Romany / Tsigane: (NEEDED)
Romanian: Pot să mănânc sticlă și ea nu mă rănește.
Esperanto: Mi povas manĝi vitron, ĝi ne damaĝas min.
Pictish: (NEEDED)
Breton: (NEEDED)
Cornish: Mý a yl dybry gwéder hag éf ny wra ow ankenya.
Welsh: Dw i'n gallu bwyta gwydr, 'dyw e ddim yn gwneud dolur i mi.
Manx Gaelic: Foddym gee glonney agh cha jean eh gortaghey mee.
Old Irish (Ogham): ᚛᚛ᚉᚑᚅᚔᚉᚉᚔᚋ ᚔᚈᚔ ᚍᚂᚐᚅᚑ ᚅᚔᚋᚌᚓᚅᚐ᚜
Old Irish (Latin): Con·iccim ithi nglano. Ním·géna.
Irish: Is féidir liom gloinne a ithe. Ní dhéanann sí dochar ar bith dom.
Ulster Gaelic: Ithim-sa gloine agus ní miste damh é.
Scottish Gaelic: S urrainn dhomh gloinne ithe; cha ghoirtich i mi.
Anglo-Saxon (Runes): ᛁᚳ᛫ᛗᚨᚷ᛫ᚷᛚᚨᛋ᛫ᛖᚩᛏᚪᚾ᛫ᚩᚾᛞ᛫ᚻᛁᛏ᛫ᚾᛖ᛫ᚻᛖᚪᚱᛗᛁᚪᚧ᛫ᛗᛖ᛬
Anglo-Saxon (Latin): Ic mæg glæs eotan ond hit ne hearmiað me.
Middle English: Ich canne glas eten and hit hirtiþ me nouȝt.
English: I can eat glass and it doesn't hurt me.
English (IPA): [aɪ kæn iːt glɑːs ænd ɪt dɐz nɒt hɜːt miː] (Received Pronunciation)
English (Braille): ⠊⠀⠉⠁⠝⠀⠑⠁⠞⠀⠛⠇⠁⠎⠎⠀⠁⠝⠙⠀⠊⠞⠀⠙⠕⠑⠎⠝⠞⠀⠓⠥⠗⠞⠀⠍⠑
Jamaican: Mi kian niam glas han i neba hot mi.
Lalland Scots / Doric: Ah can eat gless, it disnae hurt us.
Glaswegian: (NEEDED)
Gothic (4): 𐌼𐌰𐌲 𐌲𐌻𐌴𐍃 𐌹̈𐍄𐌰𐌽, 𐌽𐌹 𐌼𐌹𐍃 𐍅𐌿 𐌽𐌳𐌰𐌽 𐌱𐍂𐌹𐌲𐌲𐌹𐌸.
Old Norse (Runes): ᛖᚴ ᚷᛖᛏ ᛖᛏᛁ ᚧ ᚷᛚᛖᚱ ᛘᚾ ᚦᛖᛋᛋ ᚨᚧ ᚡᛖ ᚱᚧᚨ ᛋᚨᚱ
Old Norse (Latin): Ek get etið gler án þess að verða sár.
Norsk / Norwegian (Nynorsk): Eg kan eta glas utan å skada meg.
Norsk / Norwegian (Bokmål): Jeg kan spise glass uten å skade meg.
Føroyskt / Faroese: Eg kann eta glas, skaðaleysur.
Íslenska / Icelandic: Ég get etið gler án þess að meiða mig.
Svenska / Swedish: Jag kan äta glas utan att skada mig.
Dansk / Danish: Jeg kan spise glas, det gør ikke ondt på mig.
Sønderjysk: Æ ka æe glass uhen at det go mæ naue.
Frysk / Frisian: Ik kin glês ite, it docht me net sear.
Nederlands / Dutch: Ik kan glas eten, het doet mĳ geen kwaad.
Kirchröadsj/Bôchesserplat: Iech ken glaas èèse, mer 't deet miech jing pieng.
Afrikaans: Ek kan glas eet, maar dit doen my nie skade nie.
Lëtzebuergescht / Luxemburgish: Ech kan Glas iessen, daat deet mir nët wei.
Deutsch / German: Ich kann Glas essen, ohne mir zu schaden.
Ruhrdeutsch: Ich kann Glas verkasematuckeln, ohne dattet mich wat jucken tut.
Langenfelder Platt: Isch kann Jlaas kimmeln, uuhne datt mich datt weh dääd.
Lausitzer Mundart ("Lusatian"): Ich koann Gloos assn und doas dudd merr ni wii.
Odenwälderisch: Iech konn glaasch voschbachteln ohne dass es mir ebbs daun doun dud.
Sächsisch / Saxon: 'sch kann Glos essn, ohne dass'sch mer wehtue.
Pfälzisch: Isch konn Glass fresse ohne dasses mer ebbes ausmache dud.
Schwäbisch / Swabian: I kå Glas frässa, ond des macht mr nix!
Deutsch (Voralberg): I ka glas eassa, ohne dass mar weh tuat.
Bayrisch / Bavarian: I koh Glos esa, und es duard ma ned wei.
Allemannisch: I kaun Gloos essen, es tuat ma ned weh.
Schwyzerdütsch (Zürich): Ich chan Glaas ässe, das schadt mir nöd.
Schwyzerdütsch (Luzern): Ech cha Glâs ässe, das schadt mer ned.
Plautdietsch: (NEEDED)
Hungarian: Meg tudom enni az üveget, nem lesz tőle bajom.
Suomi / Finnish: Voin syödä lasia, se ei vahingoita minua.
Sami (Northern): Sáhtán borrat lása, dat ii leat bávččas.
Erzian: Мон ярсан суликадо, ды зыян эйстэнзэ а ули.
Northern Karelian: Mie voin syvvä lasie ta minla ei ole kipie.
Southern Karelian: Minä voin syvvä st'oklua dai minule ei ole kibie.
Vepsian: (NEEDED)
Votian: (NEEDED)
Livonian: (NEEDED)
Estonian: Ma võin klaasi süüa, see ei tee mulle midagi.
Latvian: Es varu ēst stiklu, tas man nekaitē.
Lithuanian: Aš galiu valgyti stiklą ir jis manęs nežeidžia
Old Prussian: (NEEDED)
Sorbian (Wendish): (NEEDED)
Czech: Mohu jíst sklo, neublíží mi.
Slovak: Môžem jesť sklo. Nezraní ma.
Polska / Polish: Mogę jeść szkło i mi nie szkodzi.
Slovenian: Lahko jem steklo, ne da bi mi škodovalo.
Bosnian, Croatian, Montenegrin and Serbian (Latin): Ja mogu jesti staklo, i to mi ne šteti.
Bosnian, Montenegrin and Serbian (Cyrillic): Ја могу јести стакло, и то ми не штети.
Macedonian: Можам да јадам стакло, а не ме штета.
Russian: Я могу есть стекло, оно мне не вредит.
Belarusian (Cyrillic): Я магу есці шкло, яно мне не шкодзіць.
Belarusian (Lacinka): Ja mahu jeści škło, jano mne ne škodzić.
Ukrainian: Я можу їсти скло, і воно мені не зашкодить.
Bulgarian: Мога да ям стъкло, то не ми вреди.
Georgian: მინას ვჭამ და არა მტკივა.
Armenian: Կրնամ ապակի ուտել և ինծի անհանգիստ չըներ։
Albanian: Unë mund të ha qelq dhe nuk më gjen gjë.
Turkish: Cam yiyebilirim, bana zararı dokunmaz.
Turkish (Ottoman): جام ييه بلورم بڭا ضررى طوقونمز
Tatar: Алам да бар, пыяла, әмма бу ранит мине.
Uzbek / O’zbekcha: (Roman): Men shisha yeyishim mumkin, ammo u menga zarar keltirmaydi.
Uzbek / Ўзбекча (Cyrillic): Мен шиша ейишим мумкин, аммо у менга зарар келтирмайди.
Bangla / Bengali: আমি কাঁচ খেতে পারি, তাতে আমার কোনো ক্ষতি হয় না।
Marathi (masculine): मी काच खाऊ शकतो, मला ते दुखत नाही.
Marathi (feminine):   मी काच खाऊ शकते, मला ते दुखत नाही.
Kannada: ನನಗೆ ಹಾನಿ ಆಗದೆ, ನಾನು ಗಜನ್ನು ತಿನಬಹುದು
Hindi (masculine): मैं काँच खा सकता हूँ और मुझे उससे कोई चोट नहीं पहुंचती.
Hindi (feminine):   मैं काँच खा सकती हूँ और मुझे उससे कोई चोट नहीं पहुंचती.
Malayalam: എനിക്ക് ഗ്ലാസ് തിന്നാം. അതെന്നെ വേദനിപ്പിക്കില്ല.
Tamil: நான் கண்ணாடி சாப்பிடுவேன், அதனால் எனக்கு ஒரு கேடும் வராது.
Telugu: నేను గాజు తినగలను మరియు అలా చేసినా నాకు ఏమి ఇబ్బంది లేదు
Sinhalese: මට වීදුරු කෑමට හැකියි. එයින් මට කිසි හානියක් සිදු නොවේ.
Urdu(3): میں کانچ کھا سکتا ہوں اور مجھے تکلیف نہیں ہوتی ۔
Pashto(3): زه شيشه خوړلې شم، هغه ما نه خوږوي
Farsi / Persian(3): .من می توانم بدونِ احساس درد شيشه بخورم
Arabic(3): أنا قادر على أكل الزجاج و هذا لا يؤلمني.
Aramaic: (NEEDED)
Maltese: Nista' niekol il-ħġieġ u ma jagħmilli xejn.
Hebrew(3): אני יכול לאכול זכוכית וזה לא מזיק לי.
Yiddish(3): איך קען עסן גלאָז און עס טוט מיר נישט װײ.
Judeo-Arabic: (NEEDED)
Ladino: (NEEDED)
Gǝʼǝz: (NEEDED)
Amharic: (NEEDED)
Twi: Metumi awe tumpan, ɜnyɜ me hwee.
Hausa (Latin): Inā iya taunar gilāshi kuma in gamā lāfiyā.
Hausa (Ajami) (2): إِنا إِىَ تَونَر غِلَاشِ كُمَ إِن غَمَا لَافِىَا
Yoruba(4): Mo lè je̩ dígí, kò ní pa mí lára.
Lingala: Nakokí kolíya biténi bya milungi, ekosála ngáí mabé tɛ́.
(Ki)Swahili: Naweza kula bilauri na sikunyui.
Malay: Saya boleh makan kaca dan ia tidak mencederakan saya.
Tagalog: Kaya kong kumain nang bubog at hindi ako masaktan.
Chamorro: Siña yo' chumocho krestat, ti ha na'lalamen yo'.
Fijian: Au rawa ni kana iloilo, ia au sega ni vakacacani kina.
Javanese: Aku isa mangan beling tanpa lara.
Burmese (Unicode 4.0): က္ယ္ဝန္‌တော္‌၊က္ယ္ဝန္‌မ မ္ယက္‌စားနုိင္‌သည္‌။ ၎က္ရောင္‌့ ထိခုိက္‌မ္ဟု မရ္ဟိပာ။ (9)
Burmese (Unicode 5.0): ကျွန်တော် ကျွန်မ မှန်စားနိုင်တယ်။ ၎င်းကြောင့် ထိခိုက်မှုမရှိပါ။ (9)
Vietnamese (quốc ngữ): Tôi có thể ăn thủy tinh mà không hại gì.
Vietnamese (nôm) (4): 些 𣎏 世 咹 水 晶 𦓡 空 𣎏 害 咦
Khmer: ខ្ញុំអាចញុំកញ្ចក់បាន ដោយគ្មានបញ្ហារ
Lao: ຂອ້ຍກິນແກ້ວໄດ້ໂດຍທີ່ມັນບໍ່ໄດ້ເຮັດໃຫ້ຂອ້ຍເຈັບ.
Thai: ฉันกินกระจกได้ แต่มันไม่ทำให้ฉันเจ็บ
Mongolian (Cyrillic): Би шил идэй чадна, надад хортой биш
Mongolian (Classic) (5): ᠪᠢ ᠰᠢᠯᠢ ᠢᠳᠡᠶᠦ ᠴᠢᠳᠠᠨᠠ ᠂ ᠨᠠᠳᠤᠷ ᠬᠣᠤᠷᠠᠳᠠᠢ ᠪᠢᠰᠢ
Dzongkha: (NEEDED)
Nepali: ﻿म काँच खान सक्छू र मलाई केहि नी हुन्‍न् ।
Tibetan: ཤེལ་སྒོ་ཟ་ནས་ང་ན་གི་མ་རེད།
Chinese: 我能吞下玻璃而不伤身体。
Chinese (Traditional): 我能吞下玻璃而不傷身體。
Taiwanese(6): Góa ē-tàng chia̍h po-lê, mā bē tio̍h-siong.
Japanese: 私はガラスを食べられます。それは私を傷つけません。
Korean: 나는 유리를 먹을 수 있어요. 그래도 아프지 않아요
Bislama: Mi save kakae glas, hemi no save katem mi.
Hawaiian: Hiki iaʻu ke ʻai i ke aniani; ʻaʻole nō lā au e ʻeha.
Marquesan: E koʻana e kai i te karahi, mea ʻā, ʻaʻe hauhau.
Inuktitut (10): ᐊᓕᒍᖅ ᓂᕆᔭᕌᖓᒃᑯ ᓱᕋᙱᑦᑐᓐᓇᖅᑐᖓ
Chinook Jargon: Naika məkmək kakshət labutay, pi weyk ukuk munk-sik nay.
Navajo: Tsésǫʼ yishą́ągo bííníshghah dóó doo shił neezgai da.
Cherokee (and Cree, Chickasaw, Cree, Micmac, Ojibwa, Lakota, Náhuatl, Quechua, Aymara, and other American languages): (NEEDED)
Garifuna: (NEEDED)
Gullah: (NEEDED)
Lojban: mi kakne le nu citka le blaci .iku'i le se go'i na xrani mi
Nórdicg: Ljœr ye caudran créneþ ý jor cẃran.
'''

expected_tokens = [11, 6759, 46, 57, 51403, 61654, 21265, 47425, 32223, 11, 65458, 53398, 261, 23907, 2243, 3963, 1927, 295, 25623, 5385, 9086, 47, 1795, 47, 7206, 47, 1735, 47, 2236, 619, 2040, 108, 645, 612, 280, 3493, 635, 46, 628, 46, 648, 280, 3684, 3670, 287, 47, 49, 261, 24242, 32223, 30853, 21413, 31073, 22799, 51667, 45, 21835, 32515, 21061, 46, 57, 51403, 51894, 11, 27405, 8439, 40050, 4706, 51328, 45, 22066, 41440, 45, 4715, 60111, 63667, 21061, 46, 57, 11, 58430, 47, 29894, 30853, 4600, 22187, 38990, 4811, 4435, 332, 51285, 24566, 32223, 47, 3918, 30659, 11, 8260, 60235, 21273, 62293, 52382, 47, 59022, 45, 39934, 4600, 4692, 22752, 4811, 11, 35, 26651, 35, 4715, 269, 25392, 35, 32234, 32223, 30853, 45, 30788, 47544, 22590, 32224, 30659, 53057, 332, 11, 42092, 24381, 51403, 59222, 4424, 32074, 46828, 47, 20134, 21244, 4600, 45, 52028, 45, 4811, 11, 25581, 22799, 39942, 37598, 45, 21265, 32223, 45, 22590, 59222, 4706, 32515, 21061, 46, 57, 51403, 4712, 332, 11, 42343, 24752, 61689, 4706, 53238, 46297, 47, 61369, 4788, 21660, 57344, 11, 27142, 31461, 38499, 46, 27165, 50982, 4706, 21061, 46, 57, 51403, 116, 30858, 4424, 38879, 22226, 11, 49695, 52569, 4596, 39931, 51403, 40085, 32234, 30853, 47, 261, 6699, 32223, 38905, 37837, 38132, 55647, 61738, 45, 22066, 41440, 21061, 46, 57, 11, 58430, 45, 4423, 32458, 4423, 59407, 51600, 21061, 46, 57, 60483, 4706, 50762, 30469, 11, 42074, 32227, 47266, 39115, 39166, 4596, 332, 51328, 21061, 46, 57, 30853, 47, 261, 57923, 4811, 20101, 3483, 684, 55, 46, 50, 59, 640, 620, 45, 57178, 303, 47, 56, 21265, 285, 47, 52, 100, 45, 332, 45730, 11, 26827, 34922, 21061, 46, 57, 39652, 59904, 332, 269, 8147, 41440, 57195, 4596, 22590, 31937, 22752, 11, 27142, 4601, 59904, 116, 332, 59284, 32227, 4600, 52391, 22590, 50860, 47448, 35, 21265, 11, 35, 60946, 32227, 21286, 22187, 47806, 22590, 50860, 47448, 39652, 4435, 59851, 11, 2215, 22590, 32355, 35, 4450, 332, 60327, 45730, 47, 20556, 55791, 32354, 55532, 4596, 11, 6759, 46, 57, 51403, 116, 4600, 4811, 52716, 21273, 22066, 41440, 21061, 46, 57, 57195, 4450, 332, 11, 63095, 59284, 275, 86, 44, 5904, 69, 497, 40186, 38932, 332, 21365, 31296, 4419, 56494, 11, 54278, 31376, 45, 4715, 332, 52906, 47483, 47, 3918, 39021, 4435, 332, 30992, 31135, 4811, 11, 42508, 2030, 63493, 332, 22066, 41440, 21061, 46, 57, 57195, 30917, 332, 59407, 11, 49092, 50762, 59284, 32227, 4600, 31198, 22187, 59206, 4596, 22590, 51372, 11, 25442, 21400, 60111, 38566, 38881, 45, 30788, 47544, 20101, 3483, 684, 55, 46, 50, 38278, 461, 11, 26204, 7066, 32234, 47, 3913, 21273, 30405, 45, 31198, 56399, 22066, 41440, 60483, 4715, 11, 8933, 2251, 7832, 7162, 61671, 30659, 22187, 51285, 4811, 20101, 3483, 684, 55, 45, 32475, 31358, 11, 34210, 25504, 31458, 59516, 45, 21265, 21413, 31262, 4811, 32355, 59365, 47, 261, 40674, 38008, 45, 53342, 332, 22066, 41440, 21061, 46, 57, 57195, 4600, 275, 50, 42, 63841, 4424, 11, 6984, 45, 275, 51, 42, 63841, 4450, 51665, 22226, 47289, 63839, 59284, 275, 2101, 11, 61001, 47277, 497, 21265, 275, 52, 42, 22590, 59725, 60313, 31376, 37634, 4419, 11, 25794, 7600, 21061, 46, 57, 57195, 4600, 59407, 59555, 45, 340, 47, 102, 47, 46928, 11, 8593, 49880, 49344, 39888, 39291, 63613, 37634, 21273, 22066, 41440, 11, 54342, 47, 29894, 30853, 31947, 269, 6670, 19794, 35, 4596, 22590, 31254, 31300, 45, 4788, 4588, 22799, 21556, 461, 11, 8695, 32227, 45, 32515, 51403, 51347, 52951, 45285, 45, 40186, 47266, 45150, 4435, 11, 34085, 21700, 51273, 47, 261, 5559, 38905, 4596, 32234, 30853, 21286, 51665, 3552, 61671, 31323, 275, 26708, 22590, 31300, 11, 25404, 499, 3913, 55452, 45, 21256, 38905, 21616, 32487, 269, 125, 384, 45913, 21700, 22590, 22638, 32223, 11, 35009, 285, 47, 50, 47, 50, 21265, 285, 47, 51, 47, 50, 45, 40186, 51306, 22184, 46, 35433, 24381, 36225, 55857, 11, 86, 44, 620, 620, 21265, 320, 44, 620, 56, 71, 47, 3908, 22799, 51497, 32234, 30853, 32487, 332, 38501, 46, 36008, 30890, 45, 11, 35758, 269, 125, 35, 61671, 47266, 21256, 31300, 4833, 4596, 45521, 3552, 275, 35551, 46513, 499, 11, 24242, 45143, 22799, 4811, 32223, 52628, 45, 53342, 32515, 21061, 46, 57, 51403, 38492, 22590, 11, 49016, 46685, 4706, 61671, 4596, 38418, 31300, 45, 32227, 4600, 53342, 30719, 11, 8147, 41440, 60483, 4600, 57094, 4450, 332, 47289, 63839, 59284, 47, 261, 23962, 32227, 45, 4423, 4419, 63277, 4811, 22590, 46681, 4706, 22066, 41440, 57195, 32354, 11, 25586, 45, 4601, 4600, 30135, 332, 60151, 61517, 275, 7005, 4596, 32074, 62559, 30788, 11, 42092, 24381, 42, 57260, 4811, 60389, 30719, 62105, 30364, 4706, 332, 22066, 41440, 11, 54342, 32487, 332, 63839, 59284, 47, 3908, 22799, 46018, 32234, 57317, 4596, 11, 27537, 51403, 45, 32230, 46842, 46246, 22590, 269, 125, 35, 45521, 47, 3328, 23727, 30482, 22590, 39923, 59, 65510, 125, 65503, 65442, 125, 11, 50, 267, 24178, 51328, 21061, 46, 57, 32224, 65504, 125, 65503, 65442, 125, 11, 6833, 47266, 22449, 22590, 36752, 32497, 274, 8036, 2038, 448, 43914, 35, 2730, 9795, 2739, 27815, 35, 65440, 125, 65503, 65442, 125, 11, 51, 267, 53426, 59354, 32223, 37970, 65489, 125, 65503, 65442, 125, 11, 51, 47, 50, 267, 33106, 56895, 57195, 4706, 332, 51161, 46426, 65450, 125, 65503, 65442, 125, 11, 51, 47, 50, 47, 50, 267, 50, 30364, 267, 41, 86, 46, 620, 620, 620, 620, 501, 49927, 35, 19232, 35, 65481, 11, 51, 47, 50, 47, 51, 267, 51, 37936, 275, 86, 46, 620, 620, 620, 700, 501, 49927, 35, 2423, 35, 65479, 125, 11, 51, 47, 50, 47, 52, 267, 52, 37936, 275, 86, 46, 620, 620, 628, 620, 501, 49927, 35, 225, 161, 129, 35, 65479, 125, 11, 51, 47, 50, 47, 53, 267, 53, 37936, 275, 86, 46, 620, 621, 620, 620, 501, 49927, 35, 241, 145, 129, 129, 35, 65479, 125, 11, 51, 47, 50, 47, 54, 267, 54, 37936, 275, 86, 46, 620, 640, 620, 620, 501, 49927, 35, 64156, 19232, 35, 65479, 125, 11, 51, 47, 50, 47, 55, 267, 55, 37936, 275, 86, 46, 624, 620, 620, 620, 501, 49927, 35, 64156, 43902, 35, 65479, 125, 65503, 65442, 125, 11, 51, 47, 51, 267, 23853, 56895, 57195, 4706, 332, 51161, 46426, 65453, 125, 65503, 65442, 125, 11, 51, 47, 51, 47, 50, 267, 50, 30364, 267, 41, 86, 46, 620, 620, 620, 56, 71, 501, 49927, 35, 128, 35, 65481, 11, 51, 47, 51, 47, 51, 267, 51, 37936, 275, 86, 46, 620, 620, 627, 1011, 501, 49927, 35, 224, 192, 35, 65479, 125, 11, 51, 47, 51, 47, 52, 267, 52, 37936, 275, 86, 46, 620, 620, 23624, 501, 49927, 35, 3317, 192, 35, 65479, 125, 11, 51, 47, 51, 47, 53, 267, 53, 37936, 275, 86, 46, 620, 50, 23624, 71, 501, 49927, 35, 64156, 35, 65479, 125, 11, 51, 47, 51, 47, 54, 267, 54, 37936, 275, 86, 46, 623, 40472, 501, 49927, 35, 64156, 19232, 35, 65479, 125, 11, 51, 47, 51, 47, 55, 267, 55, 37936, 275, 86, 46, 56, 40472, 71, 501, 49927, 35, 64156, 43902, 35, 65479, 125, 65503, 65442, 125, 11, 51, 47, 52, 267, 33313, 55647, 61738, 65492, 125, 65503, 65442, 125, 11, 51, 47, 52, 47, 50, 267, 86, 46, 620, 620, 69, 56, 1011, 296, 4507, 292, 103, 4436, 296, 269, 238, 160, 192, 35, 65487, 125, 11, 51, 47, 52, 47, 51, 267, 86, 46, 620, 620, 70, 620, 49, 296, 4508, 3553, 3553, 296, 269, 239, 129, 129, 35, 65487, 125, 11, 51, 47, 52, 47, 52, 267, 86, 46, 620, 620, 5904, 69, 296, 4509, 4436, 4434, 296, 269, 19232, 35, 65487, 125, 11, 51, 47, 52, 47, 53, 267, 86, 46, 620, 630, 23624, 296, 337, 53, 291, 103, 4436, 4436, 296, 269, 245, 144, 192, 192, 35, 65483, 125, 11, 51, 47, 52, 47, 54, 267, 86, 46, 620, 631, 620, 620, 296, 337, 53, 3563, 3553, 3553, 296, 269, 64156, 35, 65483, 125, 65503, 65442, 125, 11, 52, 267, 6263, 41440, 60483, 65508, 125, 65503, 65442, 125, 11, 52, 47, 50, 267, 60917, 64328, 37936, 65487, 125, 65503, 65442, 125, 11, 23599, 62688, 64328, 30364, 47266, 4435, 62542, 47277, 8067, 4423, 332, 54810, 125, 11, 8147, 41440, 57195, 4706, 21902, 22251, 47, 65492, 125, 65503, 65442, 125, 11, 52, 47, 50, 47, 50, 267, 33106, 64328, 30364, 283, 121, 700, 59, 269, 19232, 35, 65477, 125, 11, 52, 47, 50, 47, 51, 267, 23853, 267, 49011, 33944, 30364, 283, 2303, 103, 59, 269, 19232, 35, 65477, 125, 65503, 65442, 125, 11, 52, 47, 50, 47, 52, 267, 51, 64328, 37936, 59, 269, 43902, 35, 65488, 125, 11, 52, 47, 50, 47, 53, 267, 52, 64328, 37936, 59, 269, 58648, 35, 65487, 125, 11, 52, 47, 50, 47, 54, 267, 53, 64328, 37936, 59, 269, 64156, 35, 65485, 125, 11, 52, 47, 50, 47, 55, 267, 54, 64328, 37936, 59, 269, 64156, 19232, 35, 65484, 125, 11, 52, 47, 50, 47, 56, 267, 55, 64328, 37936, 59, 269, 64156, 43902, 35, 65483, 125, 11, 52, 47, 50, 47, 57, 267, 56, 64328, 37936, 59, 269, 64156, 58648, 35, 65481, 125, 65503, 65442, 125, 11, 52, 47, 50, 47, 58, 267, 53684, 4706, 21256, 3537, 56895, 64328, 37936, 275, 49, 121, 700, 46, 49, 2303, 103, 501, 63123, 125, 65503, 65442, 125, 19242, 35, 64156, 64156, 64156, 64156, 65510, 125, 28352, 64156, 64156, 64156, 64156, 65510, 125, 28352, 64156, 64156, 64156, 64156, 65510, 125, 28352, 64156, 64156, 64156, 64156, 35, 65509, 125, 65503, 65442, 125, 11, 52, 47, 51, 267, 6218, 7450, 39801, 61671, 65501, 125, 65503, 65442, 125, 11, 52, 47, 51, 47, 50, 267, 5559, 3505, 38499, 37936, 4706, 285, 46, 24941, 60483, 275, 49, 2304, 49, 46, 49, 2305, 103, 497, 65389, 125, 49923, 25171, 56255, 4450, 332, 39750, 59284, 59, 65473, 125, 65503, 65442, 125, 19242, 35, 19232, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 65485, 125, 28352, 19232, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 269, 65483, 125, 65503, 65442, 125, 11, 52, 47, 51, 47, 51, 267, 5559, 3489, 38499, 37936, 4706, 286, 46, 24941, 60483, 275, 49, 2306, 49, 46, 49, 2306, 103, 497, 65389, 125, 49923, 25171, 56255, 4450, 332, 39750, 59284, 59, 65473, 125, 65503, 65442, 125, 19242, 35, 19232, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 269, 65483, 125, 65503, 65442, 125, 11, 52, 47, 51, 47, 52, 267, 5559, 291, 38499, 37936, 4706, 287, 46, 24941, 60483, 275, 49, 2307, 49, 46, 49, 2307, 56, 497, 65398, 125, 49923, 25171, 56255, 4450, 332, 39750, 59284, 59, 65473, 125, 65503, 65442, 125, 19242, 35, 19232, 23247, 23247, 23247, 23247, 23247, 23247, 23247, 269, 65509, 125, 65503, 65442, 125, 11, 52, 47, 51, 47, 53, 267, 5559, 287, 38499, 37936, 4706, 288, 46, 24941, 60483, 275, 49, 2307, 57, 46, 49, 2307, 99, 497, 65398, 125, 49923, 25171, 56255, 4450, 332, 39750, 59284, 59, 65473, 125, 65503, 65442, 125, 19242, 35, 19232, 23247, 23247, 23247, 269, 65513, 359, 65503, 65442, 125, 11, 52, 47, 51, 47, 54, 267, 5559, 285, 38499, 37936, 4706, 289, 46, 24941, 60483, 275, 49, 2307, 100, 46, 49, 2307, 101, 497, 65398, 125, 49923, 25171, 56255, 4450, 332, 39750, 59284, 59, 65473, 125, 65503, 65442, 125, 19242, 35, 19232, 23247, 269, 65513, 28360, 125, 65503, 65442, 125, 11, 52, 47, 52, 267, 53684, 116, 32487, 31254, 64328, 30364, 52243, 65445, 125, 65503, 65442, 125, 11, 5559, 37936, 4706, 4419, 62091, 57195, 47266, 4435, 47277, 8067, 4423, 332, 47289, 61274, 125, 11, 8147, 41440, 57195, 45, 340, 47, 102, 583, 22799, 47266, 22449, 31581, 332, 47289, 63839, 63123, 125, 11, 58181, 4596, 30719, 4706, 22590, 31515, 3483, 39923, 47, 275, 60816, 4423, 4596, 52829, 285, 42, 58655, 125, 65503, 65442, 125, 11, 52, 47, 52, 47, 50, 267, 51, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 44, 620, 620, 501, 28360, 35, 19232, 35, 65087, 125, 11, 52, 47, 52, 47, 51, 267, 52, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 44, 620, 620, 501, 28360, 35, 43902, 35, 65087, 125, 11, 52, 47, 52, 47, 52, 267, 53, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 44, 620, 620, 501, 28360, 35, 58648, 35, 65087, 125, 11, 52, 47, 52, 47, 53, 267, 54, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 44, 620, 620, 501, 28360, 35, 64156, 35, 65087, 125, 11, 52, 47, 52, 47, 54, 267, 55, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 44, 620, 620, 501, 28360, 35, 64156, 19232, 35, 65087, 125, 11, 52, 47, 52, 47, 55, 267, 51, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 46, 620, 620, 627, 1011, 501, 269, 19232, 35, 65087, 125, 11, 52, 47, 52, 47, 56, 267, 52, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 46, 620, 620, 23624, 501, 269, 19232, 35, 65087, 125, 11, 52, 47, 52, 47, 57, 267, 53, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 46, 620, 50, 23624, 71, 501, 269, 58648, 35, 65087, 125, 11, 52, 47, 52, 47, 58, 267, 54, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 46, 623, 40472, 501, 269, 64156, 35, 65087, 125, 11, 52, 47, 52, 47, 630, 289, 46, 24941, 57195, 32487, 31254, 30364, 52243, 275, 86, 46, 56, 40472, 71, 501, 269, 64156, 19232, 35, 65087, 125, 65503, 65442, 125, 11, 52, 47, 53, 267, 40390, 7461, 27168, 4706, 62091, 60483, 65473, 125, 65503, 65442, 125, 11, 5559, 22590, 3483, 60483, 4706, 286, 47, 52, 64299, 45, 22799, 47266, 22449, 3483, 22066, 41440, 54810, 125, 11, 58430, 37832, 47277, 8067, 59, 65504, 125, 65503, 65442, 125, 19242, 35, 64156, 64156, 64156, 64156, 64156, 64156, 64156, 19232, 35, 65512, 359, 65503, 65442, 125, 11, 52, 47, 54, 267, 6116, 49498, 37936, 65509, 125, 65503, 65442, 125, 11, 6699, 59725, 22638, 37936, 45409, 45185, 4596, 332, 51328, 21061, 46, 57, 47429, 65087, 125, 65503, 65442, 125, 11, 52, 47, 54, 47, 50, 267, 1870, 296, 269, 19232, 35, 65512, 359, 11, 52, 47, 54, 47, 51, 267, 1871, 296, 269, 19232, 35, 65512, 359, 11, 52, 47, 54, 47, 52, 267, 1870, 4533, 4534, 4534, 296, 269, 64156, 35, 65502, 125, 65503, 65442, 125, 11, 53, 267, 23984, 26170, 60483, 65509, 125, 65503, 65442, 125, 11, 6699, 59725, 60483, 21286, 22187, 22066, 41440, 59067, 4811, 22590, 46432, 4706, 58655, 125, 11, 8801, 50762, 285, 47, 49, 57291, 47, 50318, 45, 32232, 21286, 46470, 32230, 60050, 21265, 54810, 125, 11, 98, 51328, 21061, 46, 57, 51601, 4600, 22187, 50887, 4811, 52573, 32229, 47, 300, 269, 26977, 21061, 46, 57, 54810, 125, 11, 49033, 35, 47266, 47051, 32229, 31198, 31296, 22066, 41440, 60483, 21700, 22638, 64161, 125, 11, 42142, 116, 59, 275, 50, 42, 3918, 38686, 4811, 38202, 64251, 4588, 39202, 8367, 60483, 21286, 54810, 125, 11, 8260, 53190, 4423, 40091, 65275, 4706, 61671, 45, 51018, 32234, 38686, 49927, 125, 11, 2215, 32100, 56939, 31458, 52628, 47, 275, 51, 42, 29521, 26170, 60483, 52597, 65314, 125, 11, 63018, 65275, 4706, 61671, 45, 32227, 38128, 59979, 2030, 4435, 58655, 125, 11, 27369, 4811, 45387, 51743, 32227, 38008, 31581, 21700, 36225, 61671, 47, 19907, 64783, 125, 11, 54088, 45, 332, 285, 46, 24941, 51600, 31300, 30835, 275, 1201, 42, 40219, 22187, 4435, 45431, 4450, 332, 63123, 125, 11, 26150, 51336, 32227, 45600, 31581, 283, 121, 49, 98, 37936, 45, 21400, 4601, 40219, 39820, 4435, 65087, 125, 11, 58394, 4423, 332, 31300, 30835, 4450, 4419, 47670, 21061, 46, 57, 51403, 38869, 4596, 22590, 64783, 125, 11, 54238, 47, 28926, 332, 57181, 39310, 4706, 32398, 45, 36225, 64856, 4706, 21061, 46, 57, 54810, 125, 11, 58430, 38989, 30135, 45, 32227, 36225, 61671, 21286, 277, 26476, 43, 50887, 4811, 4435, 58655, 125, 11, 63096, 4450, 36225, 37936, 4596, 22590, 39438, 283, 121, 620, 46, 49, 121, 56, 103, 47, 4255, 45868, 32234, 64161, 125, 11, 41034, 4706, 36225, 64856, 45, 22681, 31581, 269, 26977, 21061, 46, 57, 51403, 116, 35, 32227, 63123, 125, 11, 42155, 39202, 8367, 21061, 46, 57, 60483, 21700, 40186, 332, 52888, 56111, 45921, 47, 58655, 125, 65503, 65442, 125, 11, 53, 47, 50, 267, 53514, 4706, 4419, 39202, 8367, 36225, 59284, 65469, 125, 65503, 65442, 125, 11, 24335, 332, 31926, 21061, 46, 57, 51403, 45, 21256, 4706, 22590, 59725, 30869, 39202, 8367, 65314, 125, 11, 65062, 116, 4706, 22590, 36225, 59284, 39708, 3415, 5368, 47266, 4435, 57066, 54810, 125, 11, 26148, 332, 22066, 41440, 21061, 46, 57, 57195, 45, 21700, 56467, 4450, 53047, 35942, 4601, 32487, 54810, 125, 11, 98, 63839, 59284, 47, 3908, 22799, 22449, 332, 39708, 37837, 45, 22799, 4491, 22187, 31059, 332, 58655, 125, 11, 26977, 21061, 46, 57, 51403, 34, 65511, 125, 65503, 65442, 125, 11, 53, 47, 50, 47, 50, 320, 44, 620, 51, 71, 296, 334, 49, 4411, 64161, 62, 269, 43902, 35, 65481, 125, 11, 53, 47, 50, 47, 51, 320, 44, 620, 51, 71, 296, 336, 49, 3553, 4411, 58655, 62, 269, 58648, 35, 65481, 125, 11, 53, 47, 50, 47, 52, 320, 44, 620, 51, 71, 296, 337, 49, 3553, 3553, 4411, 43914, 62, 269, 64156, 35, 65481, 125, 11, 53, 47, 50, 47, 53, 320, 44, 620, 51, 71, 296, 337, 57, 3553, 3553, 3553, 4411, 19250, 62, 269, 64156, 19232, 35, 65481, 125, 11, 53, 47, 50, 47, 54, 320, 44, 620, 51, 71, 296, 4531, 3553, 3553, 3553, 3553, 4411, 296, 269, 64156, 43902, 35, 65481, 125, 65503, 65442, 125, 11, 53, 47, 51, 267, 48581, 39202, 8367, 60483, 65490, 125, 65503, 65442, 125, 11, 32937, 22799, 22449, 22590, 51926, 50762, 40093, 32227, 4600, 39820, 60415, 4596, 4419, 54810, 125, 11, 26619, 26170, 57195, 4588, 63841, 32487, 22590, 38594, 46685, 4706, 37936, 47, 29894, 54810, 125, 11, 1955, 332, 55647, 32223, 21700, 31926, 21061, 46, 57, 51403, 116, 47, 19407, 30869, 61671, 47266, 49927, 125, 11, 1763, 57066, 31296, 22066, 41440, 21061, 46, 57, 60483, 47, 65471, 125, 65503, 65442, 125, 11, 53, 47, 51, 47, 50, 267, 86, 46, 620, 620, 620, 56, 71, 296, 334, 50, 4436, 64161, 62, 269, 43902, 35, 65471, 125, 11, 53, 47, 51, 47, 51, 267, 86, 46, 620, 620, 627, 1011, 296, 336, 49, 292, 103, 4436, 58655, 62, 269, 58648, 35, 65471, 125, 11, 53, 47, 51, 47, 52, 267, 86, 46, 620, 620, 23624, 296, 337, 49, 291, 103, 4436, 4436, 43914, 62, 269, 64156, 35, 65471, 125, 11, 53, 47, 51, 47, 53, 267, 86, 46, 620, 50, 23624, 71, 296, 337, 57, 3560, 4436, 4436, 4436, 19250, 62, 269, 64156, 19232, 35, 65471, 125, 11, 53, 47, 51, 47, 54, 267, 86, 46, 623, 40472, 296, 4531, 3556, 4436, 4436, 4436, 4436, 296, 269, 64156, 43902, 35, 65471, 125, 65503, 65442, 125, 11, 53, 47, 52, 267, 23984, 26170, 65161, 4706, 22590, 4055, 77, 59284, 65448, 125, 65503, 65442, 125, 11, 6699, 59725, 30869, 60483, 47266, 30135, 4435, 57066, 31296, 22066, 41440, 61274, 125, 11, 6759, 46, 57, 60483, 21265, 47266, 22187, 4435, 53190, 31296, 22590, 36225, 4055, 77, 65359, 125, 11, 58181, 47, 65513, 19250, 125, 65503, 65442, 125, 11, 53, 47, 52, 47, 50, 267, 86, 44, 620, 620, 296, 334, 49, 3553, 64161, 62, 269, 43902, 35, 65479, 125, 11, 53, 47, 52, 47, 51, 267, 86, 44, 620, 620, 296, 336, 49, 3553, 3553, 58655, 62, 269, 58648, 35, 65479, 125, 11, 53, 47, 52, 47, 52, 267, 86, 44, 620, 620, 296, 337, 49, 3553, 3553, 3553, 43914, 62, 269, 64156, 35, 65479, 125, 11, 53, 47, 52, 47, 53, 267, 86, 44, 620, 620, 296, 337, 57, 3553, 3553, 3553, 3553, 19250, 62, 269, 64156, 19232, 35, 65479, 125, 11, 53, 47, 52, 47, 54, 267, 86, 44, 620, 620, 296, 4531, 3553, 3553, 3553, 3553, 3553, 296, 269, 64156, 43902, 35, 65479, 125, 65503, 65442, 125, 11, 54, 267, 48541, 30469, 60202, 65505, 125, 65503, 65442, 125, 11, 6699, 59725, 21061, 46, 57, 60483, 47266, 4435, 57066, 31296, 22066, 41440, 65087, 125, 11, 58430, 45, 51018, 32232, 39115, 60389, 40091, 20101, 3483, 684, 55, 61671, 21265, 49927, 125, 11, 98, 21061, 46, 57, 51403, 32227, 50842, 32229, 39021, 59915, 57181, 56939, 61274, 125, 11, 41193, 24381, 4811, 39202, 8367, 21061, 46, 57, 60483, 47, 65479, 125, 65503, 65442, 125, 11, 54, 47, 50, 44883, 21061, 46, 636, 60604, 116, 65501, 125, 65503, 65442, 125, 11, 54, 47, 50, 47, 50, 267, 86, 44, 69, 700, 49, 296, 4507, 332, 49, 3553, 296, 269, 58648, 35, 65492, 125, 11, 54, 47, 50, 47, 51, 267, 86, 44, 933, 56, 71, 296, 4507, 4409, 4436, 296, 269, 58648, 35, 65492, 125, 11, 54, 47, 50, 47, 52, 267, 86, 44, 933, 700, 296, 4507, 4410, 3553, 296, 269, 58648, 35, 65492, 125, 11, 54, 47, 50, 47, 53, 267, 86, 44, 933, 1011, 296, 4507, 4411, 4436, 296, 269, 58648, 35, 65492, 125, 11, 54, 47, 50, 47, 54, 267, 86, 44, 934, 620, 296, 4507, 333, 49, 3553, 296, 269, 58648, 35, 65492, 125, 11, 54, 47, 50, 47, 55, 267, 86, 44, 937, 700, 296, 4507, 4435, 3553, 296, 269, 58648, 35, 65492, 125, 11, 54, 47, 50, 47, 56, 267, 86, 44, 937, 1011, 296, 4507, 4436, 4436, 296, 269, 58648, 35, 65492, 125, 65503, 65442, 125, 11, 54, 47, 51, 29539, 1843, 21061, 46, 636, 60604, 116, 65501, 125, 65503, 65442, 125, 11, 54, 47, 51, 47, 50, 267, 86, 44, 69, 700, 49, 320, 44, 934, 620, 296, 4507, 332, 49, 3553, 4507, 333, 49, 3553, 296, 269, 64156, 43902, 35, 65453, 125, 11, 54, 47, 51, 47, 51, 267, 86, 44, 69, 700, 49, 320, 44, 937, 1011, 296, 4507, 332, 49, 3553, 4507, 4436, 4436, 296, 269, 64156, 43902, 35, 65453, 125, 11, 54, 47, 51, 47, 52, 267, 86, 44, 933, 56, 71, 320, 44, 934, 620, 296, 4507, 4409, 4436, 4507, 333, 49, 3553, 296, 269, 64156, 43902, 35, 65453, 125, 11, 54, 47, 51, 47, 53, 267, 86, 44, 933, 56, 71, 320, 44, 937, 1011, 296, 4507, 4409, 4436, 4507, 4436, 4436, 296, 269, 64156, 43902, 35, 65453, 125, 11, 54, 47, 51, 47, 54, 267, 86, 44, 933, 700, 320, 44, 934, 620, 296, 4507, 4410, 3553, 4507, 333, 49, 3553, 296, 269, 64156, 43902, 35, 65453, 125, 11, 54, 47, 51, 47, 55, 267, 86, 44, 933, 700, 320, 44, 937, 1011, 296, 4507, 4410, 3553, 4507, 4436, 4436, 296, 269, 64156, 43902, 35, 65453, 125, 11, 54, 47, 51, 47, 56, 267, 86, 44, 933, 1011, 320, 44, 934, 620, 296, 4507, 4411, 4436, 4507, 333, 49, 3553, 296, 269, 64156, 43902, 35, 65453, 125, 11, 54, 47, 51, 47, 57, 267, 86, 44, 933, 1011, 320, 44, 937, 1011, 296, 4507, 4411, 4436, 4507, 4436, 4436, 296, 269, 64156, 43902, 35, 65453, 125, 65503, 65442, 125, 11, 54, 47, 52, 20524, 58181, 30469, 60202, 65490, 125, 65503, 65442, 125, 11, 6699, 59725, 269, 8256, 60946, 35, 21286, 269, 54310, 21700, 56480, 22681, 35, 4450, 64783, 125, 11, 64019, 45, 21265, 59067, 4811, 39174, 57527, 4706, 22590, 50762, 55341, 54810, 125, 11, 35, 42263, 39115, 4435, 63665, 101, 386, 50762, 19654, 8610, 34320, 270, 58, 51537, 22590, 63123, 125, 11, 8060, 8795, 63859, 47, 64201, 45, 39931, 56916, 4596, 56429, 21061, 46, 57, 30579, 43914, 125, 11, 7207, 47059, 332, 60211, 57181, 31887, 45, 59478, 4712, 32464, 22681, 4600, 31351, 4706, 49927, 125, 11, 35758, 38073, 64580, 47, 55018, 4706, 32139, 56480, 22681, 59, 65415, 125, 65503, 65442, 125, 262, 46, 29804, 30853, 28383, 32487, 3489, 46, 7159, 61671, 22074, 22681, 22590, 52033, 40093, 280, 50, 54810, 125, 19242, 62, 320, 44, 23624, 4811, 47277, 4419, 21616, 46, 2090, 46, 25420, 275, 5855, 42, 4715, 38392, 59354, 47, 65250, 125, 65503, 65442, 125, 262, 46, 3913, 32074, 21061, 46, 636, 60326, 45, 30469, 39310, 320, 44, 5904, 70, 39021, 53193, 332, 65250, 125, 19242, 24941, 46, 27102, 60096, 275, 2215, 51319, 51035, 21061, 46, 636, 1200, 21265, 21061, 46, 636, 865, 499, 63123, 125, 65503, 65442, 125, 11, 24335, 32139, 56480, 22681, 4706, 22184, 60946, 45, 4601, 22074, 4435, 59493, 21265, 39573, 49927, 125, 11, 2215, 37866, 39944, 30469, 46853, 4596, 21061, 46, 57, 51403, 116, 45, 4423, 32232, 47266, 39115, 63123, 125, 11, 35133, 62176, 2030, 4596, 56429, 21061, 46, 57, 30579, 45, 21265, 38128, 53193, 47670, 61274, 125, 11, 58173, 4596, 62597, 62378, 47, 65485, 125, 65503, 65442, 125, 11, 24007, 54055, 63788, 22184, 60946, 4596, 3489, 46, 7159, 64251, 59, 65250, 125, 65503, 65442, 125, 11, 54, 47, 52, 47, 50, 267, 86, 44, 5904, 70, 296, 4509, 4436, 4435, 296, 269, 3317, 191, 35, 65492, 125, 11, 54, 47, 52, 47, 51, 267, 86, 44, 23624, 296, 4509, 4436, 4436, 296, 269, 3317, 192, 35, 65492, 125, 65503, 65442, 125, 11, 33313, 22184, 60946, 59, 65510, 125, 65503, 65442, 125, 11, 54, 47, 52, 47, 52, 267, 86, 44, 1009, 69, 49, 3465, 320, 44, 1009, 971, 296, 269, 240, 184, 145, 240, 184, 146, 240, 184, 147, 240, 184, 148, 240, 184, 149, 240, 184, 150, 240, 184, 151, 240, 184, 152, 240, 184, 153, 240, 184, 154, 240, 184, 155, 240, 184, 156, 240, 184, 157, 240, 184, 158, 240, 184, 159, 240, 184, 160, 240, 184, 161, 240, 184, 162, 240, 184, 163, 240, 184, 164, 240, 184, 165, 240, 184, 166, 240, 184, 167, 240, 184, 168, 240, 184, 169, 240, 184, 170, 240, 184, 171, 240, 184, 172, 240, 184, 173, 240, 184, 174, 240, 184, 175, 240, 184, 176, 35, 125, 65503, 65442, 125, 11, 54, 47, 52, 47, 53, 267, 86, 44, 111, 5904, 70, 320, 44, 111, 23624, 275, 7579, 345, 296, 284, 585, 630, 42, 65481, 125, 65503, 65442, 125, 49923, 35, 3319, 192, 191, 3319, 192, 192, 241, 176, 192, 191, 241, 176, 192, 192, 241, 192, 192, 191, 241, 192, 192, 192, 242, 144, 192, 191, 242, 144, 192, 192, 242, 160, 192, 191, 242, 160, 192, 192, 242, 176, 192, 191, 242, 176, 192, 192, 242, 192, 192, 191, 242, 192, 192, 192, 243, 144, 192, 191, 243, 144, 192, 192, 65473, 125, 54807, 243, 160, 192, 191, 243, 160, 192, 192, 243, 176, 192, 191, 243, 176, 192, 192, 243, 192, 192, 191, 243, 192, 192, 192, 244, 144, 192, 191, 244, 144, 192, 192, 244, 160, 192, 191, 244, 160, 192, 192, 244, 176, 192, 191, 244, 176, 192, 192, 244, 192, 192, 191, 244, 192, 192, 192, 245, 144, 192, 191, 245, 144, 192, 192, 35, 65471, 125, 65503, 65442, 125, 11, 6670, 19794, 65513, 43914, 125, 3328, 6759, 46, 57, 51600, 47169, 39292, 46, 27139, 30853, 11, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 9841, 261, 23907, 2243, 3963, 1927, 326, 2661, 2034, 203, 180, 108, 2643, 116, 4631, 2666, 111, 94, 295, 25623, 5385, 9086, 47, 1795, 47, 7206, 47, 1735, 47, 2236, 619, 2040, 108, 645, 612, 22898, 3493, 622, 46, 627, 46, 645, 3684, 3670, 3328, 6699, 36225, 61709, 21061, 46, 57, 56111, 32354, 4596, 32234, 39292, 46, 27139, 30853, 11, 1955, 51412, 4596, 50762, 45, 20101, 3483, 684, 55, 46, 50, 45, 21265, 20697, 3495, 699, 47, 3328, 33554, 50762, 48, 6759, 46, 57, 45, 22799, 21413, 40228, 4596, 45842, 21265, 47342, 30469, 47539, 32139, 4423, 261, 23912, 41306, 116, 21265, 57167, 59, 19241, 9924, 304, 9948, 1814, 296, 316, 45, 267, 111, 22916, 22933, 45, 22928, 337, 41, 106, 42, 296, 33, 3011, 144, 338, 41, 106, 497, 36216, 227, 143, 168, 227, 143, 162, 227, 143, 156, 9974, 64116, 9966, 9977, 227, 143, 159, 227, 143, 165, 227, 143, 172, 65486, 33, 227, 143, 171, 227, 143, 163, 227, 143, 157, 9968, 98, 2448, 44, 99, 2449, 33, 227, 143, 160, 227, 143, 166, 227, 143, 171, 3331, 9900, 121, 9906, 9872, 59, 33, 227, 141, 137, 121, 227, 141, 138, 296, 22929, 227, 141, 139, 9909, 121, 227, 141, 140, 45, 5034, 22934, 4925, 2722, 296, 4925, 41, 2442, 2721, 22935, 5035, 497, 19250, 227, 143, 171, 227, 143, 163, 227, 143, 157, 9968, 64116, 9966, 33, 227, 143, 160, 227, 143, 166, 227, 143, 171, 65486, 33, 227, 143, 171, 227, 143, 163, 227, 143, 157, 227, 143, 184, 334, 9860, 3340, 227, 143, 160, 227, 143, 166, 227, 143, 171, 3331, 9869, 22947, 33, 9869, 9852, 22946, 33, 9874, 22946, 33, 9871, 22946, 33, 9872, 22946, 33, 9865, 45, 65363, 227, 143, 169, 227, 143, 163, 227, 143, 157, 43914, 227, 143, 160, 227, 143, 166, 227, 143, 173, 65486, 33, 227, 143, 171, 227, 143, 163, 227, 143, 157, 22933, 28360, 227, 143, 160, 227, 143, 166, 227, 143, 171, 3331, 9947, 295, 332, 22942, 333, 22943, 334, 22944, 335, 33, 3012, 171, 33, 9946, 22919, 275, 227, 160, 167, 66, 227, 160, 168, 22920, 33, 227, 160, 171, 67, 227, 160, 172, 497, 36216, 227, 143, 171, 227, 143, 163, 227, 143, 157, 33, 227, 143, 179, 28360, 227, 143, 160, 227, 143, 166, 227, 143, 171, 65486, 33, 227, 143, 171, 227, 143, 163, 227, 143, 157, 33, 227, 143, 180, 98, 3007, 178, 46, 99, 3007, 178, 227, 143, 160, 227, 143, 166, 227, 143, 171, 3331, 51, 73, 9854, 278, 314, 9854, 33, 227, 136, 141, 285, 73, 9854, 80, 45, 317, 296, 287, 47, 56, 342, 2716, 45, 33, 227, 2416, 3493, 49, 4666, 28360, 227, 143, 170, 227, 143, 164, 227, 143, 158, 106, 62, 50, 19250, 227, 143, 161, 227, 143, 167, 227, 143, 174, 261, 6212, 7657, 27076, 116, 21265, 64354, 59, 19241, 2510, 106, 33, 9419, 117, 2627, 2661, 111, 2500, 2642, 2627, 111, 2627, 109, 337, 2627, 2661, 111, 2628, 117, 9416, 33, 2627, 2190, 2643, 2184, 2661, 102, 2561, 2642, 111, 3331, 90, 326, 2661, 203, 144, 8530, 109, 2625, 111, 1685, 4350, 111, 326, 107, 2628, 111, 1685, 30072, 326, 2661, 1975, 2666, 104, 2623, 94, 261, 827, 77, 59, 19241, 471, 87, 227, 142, 180, 87, 504, 227, 142, 180, 227, 142, 181, 87, 500, 87, 9886, 45, 87, 19250, 227, 141, 184, 9886, 227, 142, 180, 9888, 227, 142, 181, 9904, 9905, 227, 139, 132, 9841, 227, 142, 143, 227, 142, 150, 227, 141, 137, 261, 23956, 115, 32325, 34519, 122, 4596, 39292, 32224, 38485, 59, 19241, 9996, 65430, 65430, 65430, 65430, 65430, 43288, 9997, 3331, 9995, 65484, 9995, 3331, 9995, 3340, 9828, 22901, 42269, 9820, 21265, 22903, 41271, 9823, 46978, 54810, 9995, 3331, 9995, 65484, 9995, 3331, 9995, 3340, 9828, 19672, 2030, 37725, 35580, 116, 59, 22903, 1589, 36195, 30259, 31077, 9823, 33, 9995, 3331, 9995, 65484, 9995, 3331, 9995, 3340, 9828, 36919, 46, 50, 37725, 35580, 21265, 45083, 116, 59, 274, 2450, 97, 267, 9995, 3331, 9995, 65484, 9995, 3331, 9995, 3340, 9828, 33, 9821, 1818, 35948, 102, 9819, 22905, 849, 54016, 2201, 1848, 24989, 9822, 43914, 9995, 3331, 9995, 65484, 9995, 3331, 9995, 3340, 9828, 22906, 45, 33, 9827, 45, 33, 9834, 45, 22907, 45, 286, 9815, 53, 45, 22898, 45, 22929, 54, 605, 54, 45, 33, 9873, 45, 22908, 36216, 9995, 3331, 9995, 65484, 9995, 3331, 9995, 3340, 9828, 36225, 47160, 32223, 59, 284, 109, 74, 125, 45, 283, 1307, 45, 291, 67, 28360, 9995, 3331, 9995, 65415, 10006, 65429, 9966, 10007, 54810, 9995, 3331, 9995, 3340, 9828, 22590, 30785, 47483, 59, 22953, 3487, 47, 715, 22911, 22953, 54810, 9995, 3331, 9995, 65415, 10009, 65429, 9966, 10008, 54810, 9995, 3331, 9998, 65430, 65430, 65430, 65430, 65430, 43288, 9999, 261, 23515, 34782, 61671, 59, 19241, 24130, 72, 2703, 205, 139, 1474, 4189, 46, 50, 45, 332, 296, 353, 205, 136, 296, 349, 2676, 45, 332, 227, 132, 146, 22950, 333, 227, 132, 146, 261, 33143, 275, 1950, 29600, 8828, 1939, 501, 19241, 6699, 36752, 45176, 59, 19241, 2710, 9788, 5036, 2733, 2745, 2737, 9793, 2726, 2745, 22893, 2736, 9794, 5052, 9790, 2733, 5043, 9795, 2744, 2727, 3331, 27843, 9802, 5051, 2736, 2721, 2728, 2729, 2735, 9802, 5052, 9790, 2733, 5052, 27833, 27815, 2737, 9791, 45, 3331, 2739, 9788, 5036, 2733, 2745, 2737, 9793, 2726, 2745, 22893, 2736, 9794, 5052, 9790, 2733, 33, 3005, 133, 2744, 2727, 3331, 27827, 9796, 5045, 9788, 5035, 9793, 2721, 32549, 2740, 2737, 9787, 27802, 5052, 9790, 5036, 9799, 47, 19241, 226, 191, 192, 2693, 2736, 226, 191, 192, 5052, 9786, 5043, 9795, 2730, 27807, 27808, 5035, 2723, 2721, 2731, 2732, 9789, 2733, 2727, 3331, 2740, 9803, 2733, 33, 226, 192, 191, 2697, 27812, 9791, 2733, 27844, 5052, 9786, 33, 9780, 2725, 2737, 9787, 3331, 27807, 9792, 5051, 9786, 2733, 5049, 2737, 9803, 27839, 22893, 2733, 2724, 2737, 27802, 2745, 2732, 9789, 2733, 2727, 3331, 2743, 2721, 9800, 2737, 2725, 45, 33, 3005, 167, 5055, 2721, 9800, 2737, 2725, 45, 33, 226, 191, 192, 2697, 27809, 2741, 2728, 2725, 27832, 9787, 34, 19241, 23672, 332, 47350, 4706, 28755, 8752, 25265, 4596, 22590, 287, 2209, 51158, 3651, 59, 19241, 2707, 9784, 2743, 9792, 5052, 2721, 9784, 2740, 9786, 32551, 2737, 9793, 27837, 2721, 27839, 9793, 5045, 27820, 5036, 2729, 2723, 2733, 3005, 190, 2739, 2730, 27802, 2733, 45, 33, 3005, 167, 33, 9773, 2733, 2724, 2737, 27804, 33, 226, 191, 192, 2693, 2728, 2727, 27818, 9800, 27820, 45, 3331, 3005, 134, 27839, 2733, 5052, 226, 191, 192, 5038, 9779, 2738, 5052, 9786, 5049, 2737, 9787, 2723, 27814, 27839, 22893, 27827, 2722, 2731, 9789, 2744, 2745, 32548, 9792, 33, 3005, 134, 27839, 2733, 5049, 2737, 9794, 2738, 32554, 9796, 2738, 3331, 2731, 9795, 2723, 27824, 2738, 5048, 3005, 148, 2738, 22893, 2730, 2735, 9797, 2745, 207, 136, 32554, 9796, 2738, 5045, 9788, 2733, 5036, 9786, 2737, 5044, 9795, 2723, 27824, 2738, 5049, 2725, 2737, 9792, 32554, 9802, 3331, 27842, 2732, 2745, 2737, 9791, 2739, 2721, 2739, 2728, 27798, 5030, 9793, 27811, 2736, 27827, 2733, 33, 9783, 2737, 9803, 5036, 2729, 2723, 27819, 2732, 9789, 27819, 2741, 2738, 45, 5052, 9786, 5037, 9788, 5049, 2737, 9787, 2723, 27814, 2740, 226, 191, 192, 3331, 2725, 9779, 2738, 32554, 9802, 27843, 5049, 27833, 9791, 2730, 27822, 27839, 45, 267, 3005, 166, 2739, 2728, 226, 191, 192, 33, 3005, 134, 2736, 2745, 2738, 5045, 9790, 5049, 27802, 2739, 9795, 27815, 2728, 226, 191, 192, 5034, 9784, 27843, 9792, 3331, 2736, 2737, 9795, 27840, 27833, 2733, 32548, 2730, 9803, 2738, 5051, 2730, 9789, 2744, 2721, 2739, 2728, 27798, 5037, 9789, 27822, 47, 5048, 9784, 2724, 9789, 2733, 5048, 3005, 151, 2733, 33, 9773, 27812, 2735, 5045, 27820, 5037, 2735, 2730, 2735, 9802, 27836, 2733, 3331, 2735, 9780, 5052, 9786, 32554, 27805, 9802, 27839, 5044, 9789, 2723, 27822, 27840, 2738, 33, 226, 189, 163, 5052, 9790, 2733, 33, 9785, 2736, 9795, 2728, 2725, 27836, 2733, 45, 5049, 2725, 2737, 9792, 33, 226, 189, 168, 2738, 5035, 27824, 27809, 9797, 2725, 2739, 2728, 27798, 45, 3331, 2735, 9784, 2743, 9792, 5052, 9790, 2733, 5048, 3005, 151, 2739, 27799, 32551, 27832, 27837, 9787, 2733, 27840, 2738, 33, 9785, 2732, 9800, 2733, 33, 226, 189, 130, 27814, 2737, 2740, 9787, 2733, 27802, 2733, 47, 22894, 2723, 3005, 189, 5037, 9789, 45, 33, 3005, 134, 27842, 5045, 9789, 2733, 3331, 27827, 2740, 226, 191, 192, 22894, 2734, 9799, 2733, 5052, 226, 192, 136, 5049, 9795, 27809, 2729, 32548, 9792, 5052, 9786, 5034, 9785, 2740, 9799, 2738, 33, 9776, 2743, 27802, 2733, 22893, 2739, 2742, 2721, 2731, 9803, 2738, 32548, 9792, 5030, 9793, 27811, 2736, 27827, 2733, 3331, 27842, 2732, 2745, 2737, 9791, 2739, 2721, 2739, 2728, 27798, 45, 32548, 9792, 5045, 9787, 2731, 226, 191, 192, 22893, 2730, 27832, 2722, 9803, 2738, 5048, 226, 189, 183, 2724, 2721, 207, 136, 22894, 2736, 226, 191, 192, 22894, 27817, 9802, 5036, 9787, 2737, 45, 5048, 9784, 5049, 9787, 27808, 2729, 3331, 2723, 9789, 2723, 27822, 27803, 5052, 2721, 9802, 2740, 226, 191, 192, 22893, 2732, 2742, 9795, 27840, 27831, 207, 136, 5046, 9802, 2733, 5045, 9789, 2733, 27843, 2729, 5049, 9789, 2736, 27802, 2739, 27814, 2729, 32554, 9802, 2728, 226, 191, 192, 33, 9780, 27807, 2733, 9794, 2733, 3331, 2736, 27833, 27808, 2722, 2725, 9800, 2733, 33, 9778, 2732, 9800, 2733, 5038, 226, 189, 183, 27818, 2729, 5052, 9790, 2733, 5049, 2737, 3005, 190, 27841, 2733, 45, 33, 3005, 134, 2736, 2745, 2738, 32554, 9796, 2738, 32552, 2732, 2732, 9787, 2743, 27824, 2738, 3331, 2739, 3005, 190, 2739, 2735, 27815, 2733, 47, 22894, 9786, 2733, 5036, 9786, 2737, 32554, 9802, 27843, 5035, 2725, 2722, 2721, 9793, 2745, 2738, 33, 9785, 2736, 9787, 2737, 2734, 226, 192, 132, 45, 5052, 9795, 27840, 32548, 9792, 5049, 2725, 2737, 9792, 32554, 9802, 3331, 2740, 9793, 27818, 5052, 2729, 2732, 2745, 2737, 9791, 2739, 2725, 27839, 9793, 5052, 2729, 2738, 32548, 9792, 33, 3005, 132, 2733, 5052, 2737, 9795, 27827, 2733, 22894, 2734, 9789, 27837, 27798, 5051, 2730, 2735, 2736, 2725, 9800, 2733, 207, 136, 5049, 2737, 9792, 2733, 5037, 9788, 3331, 2740, 9790, 2733, 22893, 2737, 2743, 9790, 2733, 33, 9782, 2737, 2728, 9803, 2738, 33, 9785, 27827, 2728, 9789, 2739, 2728, 27798, 45, 5045, 9787, 42626, 27822, 33, 9778, 2723, 2735, 9802, 27814, 2729, 5049, 2725, 2737, 9792, 5052, 9799, 2738, 3331, 27840, 27809, 2741, 2740, 9799, 2738, 33, 9783, 2733, 27842, 27819, 9802, 2733, 5049, 27820, 2725, 9800, 2739, 2728, 27798, 5044, 9795, 2723, 27822, 47, 19241, 2696, 2727, 27817, 2739, 2728, 9789, 27819, 2741, 2738, 45, 5020, 226, 2422, 33, 226, 191, 192, 2707, 2731, 2741, 2733, 2728, 27805, 2730, 9794, 2738, 261, 33132, 7745, 59, 19241, 23672, 332, 50762, 61742, 62158, 59, 19241, 9681, 9686, 9703, 9692, 9684, 9686, 33, 9679, 9703, 9689, 9679, 9684, 9683, 33, 9681, 9679, 9687, 9679, 9694, 9692, 9686, 33, 9694, 9683, 9681, 9687, 9695, 9696, 9694, 9679, 9701, 9687, 9679, 50762, 46, 9687, 9695, 33, 9690, 9683, 9679, 9686, 9683, 33, 9695, 9679, 9683, 9694, 9686, 9679, 9700, 9692, 9694, 9687, 9695, 9692, 3331, 9688, 9692, 9691, 9698, 9683, 9694, 9683, 9691, 9701, 9687, 9679, 9685, 9683, 33, 9682, 9679, 9695, 9679, 9695, 2998, 173, 9694, 43270, 9679, 9682, 45, 33, 9694, 9692, 9690, 9683, 9689, 9687, 9701, 33, 9681, 9679, 9687, 9690, 9679, 9694, 9686, 43270, 9679, 3483, 46, 632, 33, 9690, 9679, 9694, 9696, 9695, 45, 3331, 9699, 47, 33, 9690, 9679, 9687, 9691, 9701, 9700, 9687, 45, 33, 9681, 9683, 9694, 9690, 9679, 9691, 9687, 9679, 9700, 9687, 47, 33, 9688, 9692, 9691, 9698, 9683, 9694, 9683, 9691, 9701, 9687, 9679, 33, 9700, 9683, 2998, 177, 9688, 9694, 43270, 9695, 33, 9683, 9694, 9686, 9679, 9682, 33, 9690, 9695, 9692, 9698, 9689, 9687, 9692, 9695, 3331, 9683, 9699, 9695, 9693, 9683, 9694, 9696, 43270, 9695, 33, 9687, 9695, 9683, 9686, 33, 9682, 9679, 9694, 9681, 43270, 9700, 9687, 33, 9694, 9692, 9681, 9692, 9694, 9687, 9701, 9679, 9679, 33, 9687, 9691, 9696, 9683, 9694, 9691, 9683, 9696, 9687, 33, 9682, 9679, 50762, 46, 9687, 45, 3331, 9687, 9691, 9696, 9683, 9694, 9691, 9679, 9701, 9687, 9692, 9691, 9679, 9689, 9687, 9685, 9679, 9701, 9687, 9679, 33, 9682, 9679, 33, 9689, 9692, 9688, 9679, 9689, 9687, 9685, 9679, 9701, 9687, 9679, 45, 50762, 46, 9687, 9695, 33, 9681, 9679, 9690, 9692, 2998, 168, 9683, 9691, 43270, 9679, 3331, 9692, 9693, 9683, 9694, 9679, 9701, 9687, 9697, 9689, 33, 9695, 9687, 9695, 9696, 9683, 9690, 43270, 9695, 9679, 45, 33, 9682, 9679, 33, 9681, 9679, 9690, 9692, 2998, 168, 9683, 9691, 43270, 9687, 9686, 33, 9693, 9694, 9692, 9681, 9694, 9679, 9690, 43270, 9700, 9687, 45, 33, 9700, 9694, 9687, 9698, 9696, 43270, 9700, 9687, 45, 3331, 9696, 9683, 9699, 9695, 9696, 43270, 9687, 9695, 33, 9682, 9679, 9690, 9697, 9700, 9679, 9684, 43270, 9679, 9695, 9679, 33, 9682, 9679, 33, 9690, 9694, 9679, 9684, 9679, 9689, 9683, 9691, 9692, 9684, 9679, 9691, 33, 9688, 9692, 9690, 9693, 9687, 9697, 9696, 9683, 9694, 9697, 9689, 33, 9695, 9687, 9695, 9696, 9683, 9690, 43270, 9700, 9687, 47, 261, 48701, 59, 19241, 23672, 332, 50762, 61742, 62158, 59, 19241, 27850, 28045, 27901, 54721, 28049, 27971, 27929, 2822, 32771, 2803, 43200, 32739, 32581, 28069, 28077, 2824, 62745, 61174, 42962, 47888, 54764, 42963, 2824, 32756, 3331, 48771, 45, 65207, 32776, 43110, 27964, 28069, 3483, 46, 632, 62825, 3492, 717, 57678, 5093, 47895, 2807, 28108, 5093, 62743, 42940, 47, 3331, 27852, 2807, 54764, 42963, 2825, 48138, 42772, 2812, 57908, 42826, 5101, 43063, 48197, 43011, 43150, 32756, 267, 27894, 43025, 28057, 2806, 5094, 27986, 27882, 64076, 3331, 2770, 2807, 43135, 42937, 2794, 5099, 50762, 45, 32726, 42815, 27954, 2794, 43187, 5099, 62807, 28004, 61249, 54562, 43187, 45, 32679, 43016, 43246, 2802, 2824, 5099, 3331, 43024, 54585, 42948, 50762, 5093, 62885, 54790, 62847, 54775, 42967, 65225, 2815, 5099, 65220, 2806, 42967, 3331, 43024, 42881, 54512, 2815, 45, 5115, 28046, 2814, 43126, 45, 47961, 28065, 27973, 5099, 62830, 28157, 2821, 54790, 57742, 28149, 43135, 42967, 65225, 2815, 47, 261, 1497, 1741, 275, 1510, 84, 36933, 285, 501, 19241, 5898, 7498, 117, 30917, 332, 46852, 4712, 20996, 50634, 4706, 20996, 37459, 50373, 116, 275, 98, 50072, 3331, 34135, 1939, 274, 6603, 3853, 98, 5210, 19241, 1650, 65425, 5346, 2378, 65391, 5346, 94, 28352, 2993, 144, 33, 9640, 9617, 9645, 9614, 9609, 9634, 9614, 2992, 175, 9631, 9645, 9614, 9639, 9627, 43267, 9629, 9620, 9641, 9612, 9622, 9620, 9640, 9627, 9614, 9627, 9631, 9604, 9639, 9624, 9606, 267, 9618, 43262, 9616, 9601, 9639, 9601, 9625, 9601, 43263, 9615, 9638, 2993, 139, 9601, 9638, 9646, 9602, 2992, 183, 9646, 9614, 9642, 9628, 9620, 9645, 3331, 9627, 9634, 9615, 9627, 43263, 9601, 9626, 9631, 9610, 9622, 9634, 9621, 9647, 9601, 9645, 9629, 9614, 9628, 9614, 43269, 9640, 9623, 9611, 9631, 9609, 9643, 9616, 43914, 9627, 43263, 43263, 9603, 9647, 9643, 2992, 140, 9622, 9646, 9641, 9604, 9645, 9639, 9602, 9623, 9632, 9639, 9615, 9632, 9616, 9631, 9607, 9607, 9632, 28352, 9612, 9622, 9604, 9614, 9631, 9615, 9611, 9636, 9629, 9602, 9631, 9614, 9612, 9635, 9639, 9616, 9644, 9614, 58514, 9618, 2992, 183, 9645, 9604, 61274, 9615, 43269, 9614, 9639, 9620, 9636, 43263, 9605, 2992, 183, 9604, 9624, 9634, 9616, 9622, 9634, 9610, 9639, 9616, 9644, 9614, 9614, 9631, 9601, 9628, 9614, 9632, 3331, 9641, 2992, 175, 9605, 9634, 2993, 140, 9614, 9639, 9622, 9635, 9621, 9601, 9612, 9631, 9618, 9612, 9631, 9645, 9624, 9628, 9631, 9624, 9639, 9620, 9636, 43263, 9620, 9632, 54810, 9628, 9620, 9632, 9621, 9605, 9630, 2992, 135, 9645, 9632, 9620, 9609, 9606, 9631, 9645, 9624, 9610, 9631, 9624, 9627, 9633, 9603, 9631, 9607, 28352, 9639, 9628, 9620, 9636, 9629, 9614, 9602, 9631, 9615, 9643, 9627, 9643, 9623, 9645, 9639, 9627, 9636, 9629, 9605, 9632, 9601, 9639, 9603, 9628, 9632, 36216, 9622, 9631, 9615, 9628, 9620, 9632, 9616, 9645, 9632, 9639, 9602, 43269, 9620, 9632, 9639, 9623, 9621, 9629, 9632, 9627, 9631, 9607, 3331, 2992, 158, 9645, 9632, 9621, 9629, 9646, 43263, 9629, 9637, 9646, 9614, 9621, 9637, 9640, 9621, 9601, 9642, 9628, 9646, 9640, 9610, 9601, 9601, 9631, 9614, 58655, 9642, 9606, 9646, 9627, 9632, 9624, 9614, 9631, 9646, 9614, 9639, 9616, 9644, 9614, 9606, 9614, 9624, 9614, 9606, 43267, 9614, 9606, 9624, 9614, 9642, 9605, 28352, 9618, 9623, 9631, 9614, 9623, 9634, 2992, 138, 9637, 9621, 9601, 9637, 9621, 9601, 9635, 9601, 9623, 9631, 9615, 9601, 9645, 9629, 9639, 9628, 9610, 9637, 58655, 9606, 9645, 43264, 9629, 9632, 9639, 9618, 9625, 9605, 9622, 9634, 9604, 9628, 9614, 9632, 2992, 160, 43269, 9622, 9646, 43263, 9643, 9628, 9646, 3331, 9610, 9646, 43263, 9622, 9615, 9622, 9632, 2992, 135, 9645, 9632, 2992, 160, 9631, 9614, 9605, 9614, 9615, 9622, 9622, 9623, 9631, 9621, 61274, 2992, 165, 2993, 134, 9628, 9632, 9642, 9603, 9622, 9603, 9646, 9633, 9606, 9638, 9601, 9638, 9646, 9615, 9622, 9622, 9623, 9631, 9604, 9601, 9647, 33, 2992, 176, 19241, 41, 6699, 37599, 4600, 332, 22638, 46, 41186, 32224, 47, 3908, 59313, 61671, 21286, 51892, 3331, 49016, 2030, 45, 22590, 38905, 4706, 22590, 47205, 45521, 47266, 4435, 50881, 32487, 22590, 3331, 125, 59284, 37599, 580, 261, 5897, 7875, 7745, 59, 19241, 24057, 7488, 116, 4596, 22590, 3636, 7668, 1939, 56529, 59, 19241, 226, 137, 177, 226, 137, 156, 226, 140, 174, 33, 226, 139, 161, 226, 140, 174, 226, 138, 180, 226, 137, 169, 226, 137, 182, 33, 226, 139, 150, 226, 141, 138, 226, 137, 166, 33, 226, 139, 161, 226, 140, 174, 226, 139, 169, 226, 137, 177, 226, 137, 182, 226, 142, 163, 3331, 226, 138, 166, 226, 137, 140, 33, 226, 139, 172, 226, 137, 137, 226, 139, 158, 33, 226, 139, 166, 226, 139, 150, 226, 140, 177, 226, 139, 161, 226, 138, 164, 226, 138, 181, 33, 226, 138, 161, 226, 138, 135, 226, 2415, 226, 141, 161, 226, 139, 158, 226, 142, 163, 3331, 226, 141, 141, 226, 141, 166, 33, 226, 140, 172, 226, 137, 137, 226, 138, 165, 226, 138, 178, 33, 226, 138, 130, 226, 137, 158, 226, 141, 166, 226, 139, 148, 33, 226, 139, 145, 226, 140, 142, 226, 142, 163, 3331, 226, 140, 177, 226, 137, 129, 33, 226, 138, 161, 226, 137, 150, 226, 137, 142, 226, 137, 154, 33, 226, 138, 134, 226, 138, 165, 33, 226, 138, 164, 226, 140, 174, 226, 141, 161, 226, 141, 164, 33, 226, 139, 150, 226, 141, 164, 226, 138, 182, 33, 226, 138, 161, 226, 141, 137, 226, 140, 177, 226, 137, 137, 226, 140, 142, 226, 142, 163, 3331, 226, 140, 169, 226, 139, 161, 226, 142, 142, 33, 226, 140, 137, 226, 137, 137, 226, 137, 158, 226, 138, 180, 33, 226, 138, 161, 226, 138, 134, 226, 138, 165, 33, 226, 139, 161, 226, 140, 174, 226, 138, 180, 226, 137, 190, 226, 137, 158, 226, 142, 163, 3331, 226, 139, 161, 226, 140, 174, 226, 141, 166, 33, 226, 138, 161, 226, 138, 161, 226, 137, 140, 33, 226, 140, 180, 226, 140, 140, 33, 226, 138, 177, 226, 2415, 226, 138, 180, 226, 142, 163, 3331, 226, 137, 179, 226, 138, 177, 226, 137, 169, 226, 141, 138, 226, 137, 154, 33, 226, 140, 174, 226, 140, 177, 226, 137, 169, 226, 141, 142, 226, 137, 154, 226, 142, 163, 3331, 226, 138, 129, 226, 137, 182, 33, 226, 138, 161, 226, 138, 129, 226, 137, 182, 226, 142, 166, 33, 226, 140, 150, 226, 139, 150, 226, 138, 130, 226, 137, 140, 226, 137, 142, 33, 226, 138, 161, 226, 139, 166, 226, 141, 142, 226, 137, 170, 33, 226, 140, 174, 226, 137, 133, 226, 140, 180, 226, 137, 142, 226, 142, 163, 3331, 226, 140, 182, 226, 137, 174, 33, 226, 138, 163, 226, 140, 172, 226, 138, 166, 226, 137, 174, 33, 226, 139, 161, 226, 139, 150, 226, 138, 161, 226, 137, 180, 33, 226, 140, 172, 226, 137, 182, 226, 137, 174, 226, 142, 163, 3331, 226, 137, 177, 226, 140, 142, 33, 226, 139, 166, 226, 139, 150, 226, 140, 177, 226, 138, 165, 226, 138, 178, 33, 226, 139, 166, 226, 139, 150, 226, 141, 134, 33, 226, 139, 166, 226, 139, 150, 226, 140, 177, 33, 226, 141, 138, 226, 137, 169, 226, 138, 165, 226, 138, 178, 33, 226, 139, 161, 226, 140, 174, 226, 138, 177, 226, 140, 180, 226, 140, 177, 226, 137, 174, 226, 137, 158, 226, 142, 163, 3331, 226, 139, 166, 226, 141, 142, 226, 140, 157, 226, 137, 174, 33, 226, 140, 169, 226, 139, 169, 226, 142, 137, 226, 138, 177, 226, 140, 142, 226, 139, 150, 33, 226, 141, 138, 226, 137, 175, 226, 137, 175, 33, 226, 137, 180, 226, 140, 174, 226, 140, 153, 226, 141, 140, 226, 140, 142, 33, 226, 139, 161, 226, 140, 174, 226, 140, 182, 226, 137, 174, 226, 137, 158, 226, 142, 163, 3331, 226, 140, 169, 226, 141, 143, 226, 137, 169, 226, 138, 165, 226, 138, 182, 33, 226, 137, 141, 226, 138, 164, 226, 142, 166, 33, 226, 138, 163, 226, 140, 172, 226, 140, 170, 226, 138, 182, 33, 226, 140, 174, 226, 137, 182, 226, 138, 134, 33, 226, 138, 164, 226, 140, 172, 226, 140, 170, 226, 138, 182, 33, 226, 140, 172, 226, 141, 161, 226, 137, 142, 226, 138, 134, 226, 142, 163, 3331, 226, 137, 166, 226, 137, 172, 33, 226, 139, 169, 226, 2415, 226, 142, 142, 226, 138, 180, 226, 138, 182, 33, 226, 137, 142, 226, 141, 133, 226, 139, 150, 33, 226, 137, 140, 226, 142, 140, 226, 138, 180, 226, 138, 182, 226, 142, 163, 3331, 226, 140, 148, 226, 138, 164, 226, 140, 174, 33, 226, 137, 156, 226, 140, 177, 226, 137, 171, 226, 140, 172, 33, 226, 140, 169, 226, 137, 137, 226, 140, 142, 226, 142, 166, 33, 226, 141, 142, 226, 139, 150, 226, 140, 182, 33, 226, 140, 174, 226, 140, 159, 33, 226, 140, 174, 226, 140, 159, 226, 137, 172, 226, 137, 142, 226, 142, 163, 3331, 226, 140, 169, 226, 139, 166, 226, 137, 182, 226, 137, 140, 226, 137, 158, 33, 226, 139, 161, 226, 141, 137, 226, 137, 170, 33, 226, 2415, 226, 139, 172, 33, 226, 140, 169, 226, 139, 161, 226, 137, 159, 226, 137, 172, 33, 226, 139, 161, 226, 141, 137, 226, 137, 170, 33, 226, 140, 140, 226, 137, 174, 226, 139, 172, 226, 142, 163, 3331, 226, 138, 177, 226, 139, 150, 226, 141, 140, 226, 137, 143, 33, 226, 138, 163, 226, 138, 177, 226, 142, 138, 33, 226, 138, 177, 226, 2415, 226, 137, 142, 226, 137, 183, 33, 226, 138, 164, 226, 142, 138, 226, 142, 163, 3331, 226, 140, 137, 226, 140, 180, 226, 141, 134, 226, 137, 134, 33, 226, 137, 156, 226, 137, 174, 33, 226, 138, 163, 226, 137, 135, 226, 139, 150, 33, 226, 141, 169, 226, 137, 174, 226, 137, 182, 226, 137, 134, 33, 226, 139, 161, 226, 138, 182, 226, 137, 140, 226, 137, 177, 226, 140, 142, 226, 142, 163, 3331, 226, 139, 166, 226, 141, 142, 226, 137, 174, 226, 137, 134, 226, 139, 150, 33, 226, 138, 161, 226, 142, 142, 226, 137, 172, 226, 137, 190, 226, 137, 134, 33, 226, 137, 142, 226, 139, 174, 33, 226, 140, 153, 226, 137, 174, 226, 141, 140, 226, 142, 163, 261, 6549, 1858, 59, 19241, 226, 155, 188, 226, 156, 151, 33, 226, 155, 180, 226, 155, 186, 226, 155, 172, 226, 155, 167, 33, 226, 155, 167, 226, 155, 172, 226, 156, 144, 33, 226, 155, 188, 226, 156, 151, 33, 226, 156, 147, 226, 155, 163, 226, 156, 159, 226, 156, 151, 33, 226, 155, 170, 226, 155, 191, 33, 226, 155, 167, 226, 155, 172, 226, 156, 152, 33, 226, 156, 155, 226, 155, 171, 226, 155, 191, 226, 156, 159, 226, 156, 151, 33, 226, 155, 191, 226, 155, 170, 226, 155, 178, 226, 155, 167, 226, 155, 186, 226, 156, 151, 226, 155, 171, 226, 155, 178, 226, 156, 159, 226, 155, 163, 226, 156, 152, 33, 226, 155, 186, 226, 156, 130, 226, 155, 167, 33, 226, 155, 167, 226, 155, 171, 33, 226, 155, 186, 226, 156, 151, 226, 156, 166, 226, 155, 172, 19241, 41, 6392, 50199, 45, 40186, 47604, 8605, 1843, 31162, 36919, 39448, 274, 1095, 4472, 1737, 2209, 32227, 4569, 3331, 7188, 102, 22589, 1852, 31247, 102, 39136, 27463, 1834, 110, 32487, 22589, 30030, 2176, 102, 578, 21265, 38989, 274, 1095, 31929, 3331, 27142, 4569, 38917, 4596, 22590, 56715, 31247, 31503, 22590, 50811, 20866, 5357, 261, 5649, 25794, 59, 19241, 227, 162, 141, 227, 161, 130, 227, 161, 168, 227, 161, 146, 33, 227, 161, 189, 227, 161, 130, 227, 161, 147, 267, 227, 162, 142, 227, 161, 157, 227, 161, 136, 227, 161, 146, 227, 161, 186, 227, 161, 177, 227, 161, 143, 33, 227, 162, 164, 227, 161, 150, 227, 161, 141, 19241, 227, 162, 142, 227, 161, 157, 227, 161, 136, 227, 161, 146, 227, 161, 186, 33, 227, 161, 187, 227, 161, 130, 227, 161, 143, 33, 227, 161, 154, 227, 161, 146, 227, 161, 130, 227, 161, 154, 227, 161, 147, 33, 227, 161, 159, 227, 161, 150, 33, 227, 161, 132, 227, 161, 146, 227, 161, 156, 227, 161, 149, 33, 227, 161, 187, 227, 161, 139, 227, 161, 186, 227, 161, 179, 33, 227, 162, 186, 227, 161, 188, 227, 161, 146, 33, 227, 161, 139, 227, 161, 143, 33, 227, 161, 158, 227, 161, 150, 33, 227, 161, 154, 227, 161, 180, 227, 161, 132, 227, 161, 159, 3331, 227, 161, 178, 227, 161, 130, 227, 161, 159, 227, 161, 146, 227, 161, 168, 227, 161, 188, 33, 227, 161, 130, 227, 161, 132, 227, 161, 180, 227, 161, 159, 33, 227, 161, 186, 227, 161, 130, 227, 161, 159, 227, 161, 179, 33, 227, 162, 186, 227, 161, 146, 33, 227, 161, 152, 227, 161, 146, 227, 161, 156, 227, 161, 139, 227, 161, 141, 227, 161, 188, 33, 227, 161, 150, 227, 161, 140, 33, 227, 161, 154, 227, 161, 139, 227, 161, 143, 33, 227, 161, 132, 227, 161, 166, 227, 161, 152, 227, 161, 139, 227, 161, 130, 227, 161, 136, 33, 227, 161, 187, 227, 161, 130, 227, 161, 143, 3331, 227, 161, 143, 227, 161, 139, 227, 161, 156, 227, 161, 158, 227, 161, 172, 33, 227, 161, 132, 227, 161, 186, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 139, 227, 161, 136, 227, 161, 188, 227, 161, 156, 227, 161, 186, 227, 161, 142, 227, 161, 130, 227, 161, 158, 227, 161, 131, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 139, 227, 161, 136, 227, 161, 188, 227, 161, 134, 227, 161, 131, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 166, 227, 161, 158, 227, 161, 154, 227, 161, 188, 227, 161, 159, 227, 161, 130, 227, 161, 134, 227, 161, 188, 227, 161, 131, 3331, 227, 161, 130, 227, 161, 158, 227, 161, 154, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 162, 227, 161, 139, 227, 161, 146, 227, 161, 140, 33, 227, 161, 142, 227, 161, 180, 227, 161, 152, 227, 161, 158, 227, 161, 188, 227, 161, 179, 33, 227, 162, 143, 227, 161, 139, 227, 161, 152, 227, 161, 150, 227, 161, 150, 227, 161, 156, 227, 161, 146, 33, 227, 161, 143, 227, 161, 139, 227, 161, 156, 227, 161, 158, 227, 161, 172, 33, 227, 161, 139, 227, 161, 159, 227, 161, 179, 33, 227, 162, 130, 227, 161, 158, 227, 161, 154, 3331, 227, 162, 143, 227, 161, 139, 227, 161, 152, 227, 161, 150, 227, 161, 150, 227, 161, 156, 227, 161, 146, 227, 161, 177, 227, 161, 143, 33, 227, 161, 158, 227, 161, 130, 227, 161, 142, 227, 161, 146, 33, 227, 161, 187, 227, 161, 130, 227, 161, 143, 33, 227, 161, 156, 227, 161, 150, 227, 161, 150, 227, 161, 154, 33, 227, 161, 166, 227, 161, 144, 227, 161, 150, 227, 161, 158, 33, 227, 161, 177, 227, 162, 162, 227, 161, 130, 227, 161, 158, 227, 161, 156, 227, 161, 146, 227, 161, 131, 33, 227, 161, 140, 227, 161, 150, 227, 161, 152, 33, 227, 161, 130, 227, 161, 158, 227, 161, 186, 227, 161, 186, 227, 161, 149, 227, 161, 156, 33, 227, 161, 154, 227, 161, 146, 3331, 227, 161, 162, 227, 161, 150, 227, 161, 143, 227, 161, 146, 33, 227, 161, 159, 227, 161, 150, 33, 227, 161, 144, 227, 161, 166, 227, 161, 159, 33, 227, 161, 154, 227, 161, 139, 227, 161, 143, 33, 227, 161, 154, 227, 161, 130, 227, 161, 158, 227, 161, 154, 33, 227, 161, 159, 227, 161, 150, 227, 161, 179, 19241, 227, 162, 150, 227, 161, 136, 227, 161, 154, 33, 227, 162, 142, 227, 161, 157, 227, 161, 136, 227, 161, 146, 227, 161, 186, 33, 227, 161, 187, 227, 161, 130, 227, 161, 143, 33, 227, 161, 130, 227, 161, 143, 33, 227, 161, 154, 227, 161, 146, 227, 161, 130, 227, 161, 154, 33, 227, 161, 130, 227, 161, 143, 33, 227, 161, 130, 33, 227, 161, 154, 227, 161, 150, 227, 161, 150, 227, 161, 152, 227, 161, 165, 227, 161, 158, 227, 161, 130, 227, 161, 139, 227, 161, 136, 227, 161, 179, 19241, 227, 162, 142, 227, 161, 149, 227, 161, 154, 227, 161, 151, 33, 227, 162, 139, 33, 227, 161, 154, 227, 161, 150, 227, 161, 158, 227, 161, 177, 227, 161, 159, 33, 227, 161, 142, 227, 161, 146, 227, 161, 130, 227, 161, 158, 33, 227, 161, 159, 227, 161, 150, 33, 227, 161, 143, 227, 161, 130, 227, 161, 186, 33, 227, 161, 186, 227, 161, 130, 227, 161, 159, 33, 227, 162, 139, 33, 227, 161, 134, 227, 161, 158, 227, 161, 171, 227, 161, 131, 33, 227, 161, 150, 227, 161, 140, 33, 227, 161, 142, 227, 161, 186, 3331, 227, 161, 171, 227, 161, 158, 33, 227, 161, 134, 227, 161, 158, 227, 161, 171, 227, 161, 136, 227, 161, 172, 227, 161, 156, 227, 161, 146, 227, 161, 131, 33, 227, 161, 178, 227, 161, 130, 227, 161, 159, 33, 227, 161, 186, 227, 161, 188, 227, 161, 146, 33, 227, 161, 139, 227, 161, 143, 33, 227, 161, 144, 227, 161, 157, 227, 161, 159, 227, 161, 139, 227, 161, 139, 227, 161, 166, 227, 161, 136, 227, 161, 157, 227, 161, 136, 227, 161, 186, 33, 227, 161, 154, 227, 161, 146, 227, 161, 130, 227, 161, 154, 33, 227, 161, 130, 227, 161, 132, 227, 161, 180, 227, 161, 159, 3331, 227, 161, 130, 33, 227, 161, 154, 227, 161, 150, 227, 161, 150, 227, 161, 152, 227, 161, 165, 227, 161, 158, 227, 161, 130, 227, 161, 139, 227, 161, 136, 227, 161, 179, 33, 227, 162, 139, 33, 227, 161, 142, 227, 161, 139, 227, 161, 164, 227, 161, 159, 33, 227, 161, 154, 227, 161, 130, 227, 161, 168, 227, 161, 146, 33, 227, 161, 132, 227, 161, 146, 227, 161, 179, 33, 227, 161, 149, 227, 161, 139, 227, 161, 136, 227, 161, 149, 227, 161, 172, 227, 161, 131, 33, 227, 161, 142, 227, 161, 186, 227, 161, 143, 227, 161, 146, 227, 161, 136, 227, 161, 140, 227, 161, 131, 33, 227, 161, 159, 227, 161, 150, 3331, 227, 161, 152, 227, 161, 146, 227, 161, 156, 227, 161, 157, 227, 161, 154, 33, 227, 161, 130, 33, 227, 161, 139, 227, 161, 150, 227, 161, 140, 227, 161, 140, 227, 161, 149, 227, 161, 165, 227, 161, 158, 227, 161, 130, 227, 161, 139, 227, 161, 136, 33, 227, 161, 130, 227, 161, 143, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 154, 227, 161, 146, 227, 161, 130, 227, 161, 154, 227, 161, 146, 227, 161, 141, 33, 227, 161, 144, 227, 161, 139, 227, 161, 146, 227, 161, 139, 227, 161, 146, 33, 227, 161, 150, 227, 161, 140, 33, 227, 161, 139, 227, 161, 152, 227, 161, 150, 227, 161, 158, 227, 161, 142, 227, 161, 150, 227, 161, 158, 227, 161, 156, 227, 161, 188, 227, 161, 186, 3331, 227, 161, 149, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 159, 227, 161, 152, 227, 161, 130, 227, 161, 154, 227, 161, 146, 227, 161, 179, 33, 227, 162, 132, 227, 161, 166, 227, 161, 159, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 187, 227, 161, 139, 227, 161, 143, 227, 161, 154, 227, 161, 150, 227, 161, 142, 33, 227, 161, 150, 227, 161, 140, 33, 227, 161, 180, 227, 161, 152, 33, 227, 161, 130, 227, 161, 158, 227, 161, 139, 227, 161, 146, 227, 161, 141, 227, 161, 150, 227, 161, 152, 227, 161, 143, 3331, 227, 161, 139, 227, 161, 143, 33, 227, 161, 149, 33, 227, 161, 186, 227, 161, 146, 33, 227, 161, 143, 227, 161, 139, 227, 161, 142, 227, 161, 139, 227, 161, 136, 227, 161, 146, 227, 161, 135, 33, 227, 161, 130, 227, 161, 158, 227, 161, 154, 33, 227, 161, 142, 227, 161, 186, 33, 227, 161, 166, 227, 161, 158, 227, 161, 154, 227, 161, 130, 227, 161, 136, 227, 161, 136, 227, 161, 171, 227, 161, 172, 33, 227, 161, 154, 227, 161, 130, 227, 161, 158, 227, 161, 154, 227, 161, 143, 3331, 227, 161, 170, 227, 161, 130, 227, 161, 136, 227, 161, 136, 33, 227, 161, 158, 227, 161, 150, 227, 161, 159, 33, 227, 161, 154, 227, 161, 139, 227, 161, 141, 227, 161, 166, 227, 161, 152, 227, 161, 132, 33, 227, 161, 139, 227, 161, 159, 227, 161, 131, 33, 227, 161, 150, 227, 161, 152, 33, 227, 161, 186, 227, 161, 146, 33, 227, 162, 139, 227, 161, 180, 227, 161, 158, 227, 161, 159, 227, 161, 152, 227, 161, 186, 227, 161, 177, 227, 161, 143, 33, 227, 161, 154, 227, 161, 150, 227, 161, 158, 227, 161, 146, 33, 227, 161, 140, 227, 161, 150, 227, 161, 152, 227, 161, 179, 33, 227, 162, 186, 227, 161, 180, 3331, 227, 161, 187, 227, 161, 139, 227, 161, 136, 227, 161, 136, 33, 227, 161, 186, 227, 161, 188, 227, 161, 146, 227, 161, 140, 227, 161, 150, 227, 161, 152, 227, 161, 146, 33, 227, 161, 144, 227, 161, 188, 227, 161, 142, 227, 161, 139, 227, 161, 159, 33, 227, 161, 142, 227, 161, 146, 33, 227, 161, 159, 227, 161, 150, 33, 227, 161, 152, 227, 161, 146, 227, 161, 144, 227, 161, 146, 227, 161, 130, 227, 161, 159, 227, 161, 131, 33, 227, 161, 146, 227, 161, 142, 227, 161, 144, 227, 161, 154, 227, 161, 130, 227, 161, 159, 227, 161, 139, 227, 161, 139, 227, 161, 130, 227, 161, 136, 227, 161, 136, 227, 161, 186, 227, 161, 131, 33, 227, 161, 186, 227, 161, 130, 227, 161, 159, 3331, 227, 162, 142, 227, 161, 157, 227, 161, 136, 227, 161, 146, 227, 161, 186, 33, 227, 161, 187, 227, 161, 130, 227, 161, 143, 33, 227, 161, 130, 227, 161, 143, 33, 227, 161, 154, 227, 161, 146, 227, 161, 130, 227, 161, 154, 33, 227, 161, 130, 227, 161, 143, 33, 227, 161, 130, 33, 227, 161, 154, 227, 161, 150, 227, 161, 150, 227, 161, 152, 227, 161, 165, 227, 161, 158, 227, 161, 130, 227, 161, 139, 227, 161, 136, 227, 161, 179, 19241, 41, 6699, 38499, 45602, 4706, 62290, 4706, 269, 66, 58743, 36442, 35, 4450, 28766, 7472, 42, 261, 23518, 6909, 30890, 60470, 51668, 32224, 59, 19241, 5449, 5764, 1046, 1113, 1176, 1238, 1318, 1386, 1445, 1526, 1587, 90, 91, 282, 621, 643, 665, 687, 709, 3331, 40881, 1898, 7819, 2019, 8257, 2145, 8758, 2271, 9109, 4918, 2439, 2451, 2462, 2468, 2484, 2492, 2493, 2503, 2516, 2525, 3331, 9815, 9816, 9819, 9822, 9823, 9824, 9826, 9828, 9830, 9834, 9873, 2578, 2586, 2599, 2605, 9863, 5019, 2694, 2695, 2696, 2716, 2721, 2722, 2723, 2724, 2745, 5062, 2763, 2764, 2765, 2766, 27862, 2796, 2797, 2798, 3331, 9900, 9901, 9906, 9872, 9919, 9922, 9933, 9915, 22915, 9892, 3010, 169, 3010, 188, 227, 136, 164, 33, 9977, 9992, 9996, 3014, 153, 10017, 10029, 10044, 10045, 33, 19102, 19232, 227, 146, 129, 9854, 226, 189, 161, 3001, 131, 212, 166, 3003, 133, 2622, 2666, 227, 142, 143, 2891, 2853, 9679, 261, 5983, 34415, 116, 4596, 53265, 59946, 59, 19241, 33155, 40213, 45, 5024, 2721, 27810, 2732, 9789, 27831, 5043, 9795, 2739, 27815, 45, 33, 43466, 10204, 10196, 10207, 261, 5647, 51528, 59114, 39923, 59, 65484, 10015, 65503, 65363, 3015, 138, 3331, 9996, 43288, 10003, 43288, 9997, 267, 9974, 43285, 9986, 43285, 9977, 267, 10006, 43285, 9986, 43285, 10007, 267, 10006, 43285, 9986, 43285, 10007, 267, 9976, 43286, 9989, 43286, 9979, 267, 3013, 143, 3013, 147, 9976, 9978, 3340, 3014, 184, 267, 3014, 188, 33, 9976, 3013, 176, 9979, 33, 9974, 3013, 177, 9977, 19250, 3015, 139, 33, 3014, 178, 10010, 3014, 178, 10010, 3014, 180, 3014, 180, 3014, 180, 3331, 9995, 9974, 9966, 3014, 169, 9966, 9977, 9995, 267, 9968, 9996, 9994, 3014, 168, 9994, 9997, 9968, 267, 9968, 3014, 147, 9994, 3014, 171, 9994, 3014, 150, 9968, 267, 9968, 3014, 148, 9966, 3014, 130, 9966, 3014, 151, 9968, 267, 9969, 9974, 9966, 3014, 131, 9966, 9977, 9969, 267, 9981, 3014, 132, 3014, 133, 3013, 154, 267, 3014, 183, 9992, 3014, 181, 3014, 187, 3014, 140, 3014, 185, 3013, 161, 9992, 3013, 169, 33, 3013, 158, 3014, 140, 3013, 166, 19250, 3015, 140, 33, 10010, 3014, 178, 10010, 3014, 178, 3014, 180, 3014, 180, 3014, 180, 3331, 9995, 9968, 10010, 33, 3014, 178, 9968, 9995, 267, 9968, 9995, 3340, 9995, 9968, 267, 9968, 9968, 22953, 22953, 9968, 267, 9968, 9995, 33, 9969, 33, 9995, 9968, 267, 9969, 9968, 33, 3014, 192, 22953, 9969, 267, 9975, 3014, 134, 3014, 135, 9979, 3340, 3014, 182, 267, 3014, 186, 33, 9981, 3013, 184, 9983, 33, 9980, 3013, 185, 9982, 19250, 3015, 141, 33, 3014, 178, 10010, 3014, 178, 10010, 3014, 180, 3014, 180, 3014, 180, 3331, 10000, 3014, 162, 33, 3014, 180, 33, 3014, 159, 10001, 267, 9984, 3014, 163, 3340, 3014, 160, 9985, 267, 9984, 9992, 9966, 9992, 9966, 9992, 9985, 267, 9984, 3014, 172, 9966, 3014, 131, 9966, 3014, 172, 9985, 267, 3013, 164, 3013, 192, 3014, 191, 9992, 3014, 189, 3013, 192, 3013, 172, 267, 3013, 150, 9983, 3013, 151, 3013, 155, 28360, 9974, 9970, 9970, 9977, 33, 3014, 143, 33, 9976, 9971, 9971, 9979, 33, 3013, 140, 33, 10016, 33, 10010, 3014, 178, 10010, 3014, 178, 3014, 180, 3014, 180, 3014, 180, 3331, 9995, 9968, 3014, 178, 33, 10010, 9968, 9995, 267, 9968, 9995, 3340, 9995, 9968, 267, 9968, 9968, 22953, 22953, 9968, 267, 9968, 9995, 33, 9969, 33, 9995, 9968, 267, 9969, 9968, 33, 3014, 190, 22953, 9969, 267, 43292, 43293, 10019, 10019, 43291, 33, 3013, 139, 267, 3013, 135, 33, 3014, 143, 33, 3014, 144, 267, 3013, 136, 33, 3013, 140, 33, 3015, 143, 3331, 9995, 9980, 9966, 10002, 9966, 9982, 9995, 267, 9968, 9998, 9994, 3014, 165, 43289, 9968, 267, 9968, 3014, 153, 9994, 3014, 171, 9994, 3014, 156, 9968, 267, 9968, 3014, 154, 9966, 3014, 129, 9966, 3014, 157, 9968, 267, 9969, 9980, 9966, 3014, 131, 9966, 9982, 9969, 267, 43292, 43293, 10019, 10019, 43291, 33, 3013, 139, 267, 3013, 135, 33, 3014, 143, 33, 3014, 144, 267, 3013, 136, 33, 3013, 140, 33, 3015, 144, 3331, 9998, 43288, 10004, 43288, 9999, 267, 9980, 43285, 9990, 43285, 9982, 267, 10009, 43285, 9990, 43285, 10008, 267, 10009, 43285, 9990, 43285, 10008, 267, 9981, 43286, 9991, 43286, 9983, 267, 3015, 152, 10013, 3015, 151, 3015, 156, 10011, 3015, 157, 3340, 9980, 9993, 9993, 9982, 33, 3014, 143, 33, 9981, 3014, 142, 3014, 142, 9983, 33, 3013, 140, 267, 3015, 130, 10012, 3015, 132, 10013, 3015, 134, 3015, 135, 10014, 10015, 65491, 3015, 158, 10011, 3015, 153, 3015, 154, 10013, 3015, 160, 261, 24143, 26086, 59, 23243, 9461, 9487, 9464, 9458, 33, 9482, 9461, 9495, 9473, 9494, 9477, 9495, 9478, 9469, 9495, 9469, 9490, 9477, 9495, 33, 9496, 22878, 9494, 9474, 9485, 9488, 9473, 9484, 9495, 9469, 9488, 22881, 9487, 9477, 9495, 33, 2974, 166, 11, 24143, 26086, 275, 54361, 65027, 501, 21975, 1784, 9722, 5000, 1743, 8255, 2323, 2219, 2238, 60, 4692, 2111, 7697, 24727, 344, 9390, 47, 11, 48410, 1744, 36752, 59, 33, 3005, 150, 2721, 27813, 2733, 33, 2750, 2721, 2723, 2725, 9800, 2733, 5037, 9797, 27818, 27814, 2729, 207, 136, 32554, 9802, 27843, 5048, 3005, 149, 32549, 5035, 2731, 9787, 2736, 27840, 2729, 47, 11, 33143, 275, 26271, 8828, 1939, 501, 5026, 27827, 2737, 2749, 32550, 5054, 2717, 2745, 5051, 2736, 2721, 2739, 2732, 2718, 27818, 5036, 2741, 2721, 27811, 2717, 5055, 2745, 27830, 2738, 32550, 5049, 2717, 2728, 2745, 5052, 2720, 27827, 27839, 47, 11, 33143, 275, 26713, 8828, 1939, 501, 5026, 27827, 2737, 9803, 5046, 9786, 5054, 2717, 2745, 5051, 2736, 2721, 2739, 2732, 2718, 27818, 5036, 2741, 2721, 27811, 9786, 5055, 2745, 2737, 9792, 2738, 5046, 9786, 5049, 2717, 2728, 2745, 5052, 2720, 27827, 27839, 47, 11, 1001, 8663, 7207, 59, 275, 1270, 969, 969, 42, 11, 33226, 59, 21113, 8660, 4507, 7491, 31733, 2238, 60, 4663, 1922, 22184, 22180, 1859, 47, 11, 6392, 44317, 59, 3933, 31782, 31366, 7788, 21519, 32410, 2155, 47, 4063, 4660, 31546, 47, 11, 40500, 59, 3933, 22281, 121, 31366, 1857, 4497, 22700, 2155, 45, 22840, 4682, 4660, 30813, 22264, 22066, 47, 11, 24057, 34351, 109, 282, 20549, 25975, 59, 315, 2512, 1822, 22068, 7970, 4481, 4843, 7896, 45, 4660, 4678, 7582, 2159, 2495, 22264, 47, 11, 24065, 7199, 1798, 1955, 59, 309, 40, 2115, 2248, 31366, 1857, 335, 40, 2007, 22717, 2155, 45, 22840, 344, 40, 1866, 22264, 22066, 47, 11, 24316, 8379, 59, 3749, 106, 22317, 31353, 2508, 4491, 353, 2504, 2155, 45, 4968, 8435, 98, 344, 40, 21706, 2244, 22203, 111, 22129, 47, 11, 5726, 2049, 7471, 1955, 59, 275, 1270, 969, 969, 42, 11, 1222, 2167, 6957, 59, 275, 1270, 969, 969, 42, 11, 6457, 7035, 59, 3710, 40, 2115, 2248, 31429, 106, 4497, 22700, 2155, 45, 21426, 344, 40, 1880, 2503, 22090, 345, 459, 98, 47, 11, 5745, 8714, 1746, 48, 5745, 2196, 59, 275, 1270, 969, 969, 42, 11, 75, 2502, 2167, 25632, 59, 275, 1270, 969, 969, 42, 11, 1193, 1864, 2512, 109, 3647, 7921, 1853, 275, 1094, 1956, 2509, 501, 312, 9069, 21943, 22068, 1967, 353, 2502, 45, 4641, 4725, 21368, 2180, 459, 47, 11, 5623, 8549, 59, 36899, 6976, 21911, 21528, 6967, 117, 45, 4528, 21543, 31430, 108, 4516, 24742, 47, 11, 32983, 1746, 282, 36445, 2494, 59, 4138, 100, 22080, 7970, 22708, 2155, 45, 22350, 4692, 4516, 4529, 22066, 47, 11, 48727, 59, 4138, 7415, 38086, 22708, 8616, 45, 4692, 4660, 31033, 4477, 9330, 47, 11, 852, 24465, 9298, 59, 4138, 7415, 22095, 24983, 21342, 2155, 45, 4692, 4660, 40, 111, 4529, 22066, 281, 11, 852, 1746, 9298, 59, 275, 1270, 969, 969, 42, 11, 6263, 8114, 2148, 9319, 59, 275, 1270, 969, 969, 42, 11, 5967, 34609, 59, 3791, 22306, 112, 355, 24610, 115, 38151, 6959, 336, 22184, 30507, 7042, 102, 47, 11, 53510, 61445, 59, 29608, 112, 38086, 22708, 2164, 45, 31554, 4660, 21665, 22066, 47, 11, 40339, 7745, 61445, 275, 57, 501, 29608, 112, 38086, 22708, 2164, 45, 31554, 4660, 31350, 8866, 47, 11, 5715, 26619, 25132, 112, 48, 1187, 1778, 9018, 25132, 118, 275, 5720, 102, 37531, 501, 312, 40, 22306, 2504, 21480, 2504, 22708, 2170, 45, 4452, 4799, 22061, 118, 2496, 46, 110, 446, 11, 6446, 7744, 7473, 118, 59, 3636, 106, 22314, 21961, 102, 4556, 1751, 21271, 112, 336, 4692, 4799, 21800, 7846, 4477, 9330, 47, 11, 48554, 59, 29608, 112, 31366, 7747, 102, 4594, 22702, 2164, 336, 22184, 4663, 4529, 31359, 47, 11, 6284, 33785, 59, 318, 9348, 333, 9348, 4481, 31353, 2494, 4515, 22740, 7341, 45, 4515, 4660, 4529, 31429, 98, 22066, 47, 11, 33397, 59, 4025, 31733, 112, 31353, 98, 40, 4520, 22702, 2164, 45, 336, 22197, 4660, 4529, 31359, 47, 11, 1291, 26711, 8778, 112, 59, 312, 40, 22320, 123, 31353, 98, 40, 346, 5231, 2217, 45, 336, 22197, 344, 40, 4529, 22066, 47, 11, 6790, 7519, 1746, 59, 4027, 31733, 112, 31353, 7036, 4515, 22702, 2164, 45, 4692, 40, 109, 4660, 4529, 4656, 102, 47, 11, 6835, 1848, 2180, 275, 5973, 26620, 102, 501, 315, 2512, 2194, 112, 31366, 106, 2496, 346, 22695, 7382, 336, 346, 4692, 4660, 337, 2494, 344, 2496, 47, 11, 1456, 7246, 7745, 59, 4138, 8427, 118, 31366, 7747, 106, 352, 22717, 2170, 45, 22197, 4663, 4529, 22066, 106, 47, 11, 23483, 7851, 7342, 102, 275, 6604, 7356, 1937, 501, 275, 1270, 969, 969, 42, 11, 1224, 1908, 7374, 7506, 275, 6604, 7356, 1937, 501, 275, 1270, 969, 969, 42, 11, 33397, 8688, 275, 1067, 25938, 2239, 501, 3932, 118, 4774, 106, 31366, 7747, 22688, 7341, 45, 22454, 2345, 21427, 4752, 1741, 4529, 30667, 332, 22063, 47, 11, 33397, 122, 282, 4257, 25750, 102, 59, 275, 1270, 969, 969, 42, 11, 33397, 7745, 59, 20667, 22555, 344, 9396, 9254, 100, 4793, 7763, 2529, 22865, 4504, 4697, 344, 2529, 349, 9396, 102, 27787, 47, 11, 5894, 7487, 2077, 112, 59, 4027, 22318, 1751, 22068, 2549, 106, 40132, 111, 45, 33, 2549, 106, 4682, 21502, 98, 2549, 1751, 22095, 47, 11, 6457, 8817, 105, 59, 275, 1270, 969, 969, 42, 11, 5650, 8828, 59, 275, 1270, 969, 969, 42, 11, 23530, 7911, 59, 312, 2523, 332, 356, 109, 4501, 1775, 122, 4565, 36104, 4567, 104, 4970, 103, 4701, 22781, 4720, 21267, 7476, 98, 47, 11, 6812, 2183, 59, 3753, 340, 40, 111, 30939, 118, 4449, 2338, 98, 4565, 2326, 115, 45, 274, 1838, 120, 336, 4480, 1949, 356, 111, 4565, 8236, 101, 21554, 2242, 340, 4663, 47, 11, 6264, 121, 3845, 25215, 59, 3821, 1817, 2332, 4551, 102, 21764, 2072, 1864, 4412, 105, 21426, 4607, 1746, 4511, 21771, 8774, 7690, 4660, 102, 47, 11, 6392, 36843, 275, 80, 1898, 1745, 501, 33, 226, 155, 156, 226, 155, 156, 226, 155, 138, 226, 155, 146, 226, 155, 134, 226, 155, 149, 226, 155, 138, 226, 155, 138, 226, 155, 149, 226, 155, 140, 226, 155, 129, 226, 155, 149, 226, 155, 137, 226, 155, 149, 226, 155, 129, 226, 155, 142, 226, 155, 131, 226, 155, 145, 226, 155, 134, 226, 155, 146, 226, 155, 129, 226, 155, 134, 226, 155, 149, 226, 155, 140, 226, 155, 141, 226, 155, 148, 226, 155, 134, 226, 155, 145, 226, 155, 157, 11, 6392, 36843, 275, 33226, 501, 19652, 2453, 7758, 1949, 21900, 106, 4684, 8056, 112, 47, 313, 9318, 2453, 104, 9294, 98, 47, 11, 1139, 7911, 59, 3917, 21722, 7772, 115, 4641, 2097, 21764, 7861, 102, 332, 21900, 102, 47, 313, 2507, 4484, 2503, 24563, 111, 22549, 30655, 1750, 4422, 21365, 105, 21555, 47, 11, 1533, 27075, 3845, 25215, 59, 3918, 7696, 46, 2176, 21764, 7854, 4412, 2243, 345, 2507, 31438, 102, 21502, 105, 4970, 47, 11, 33421, 7911, 3845, 25215, 59, 318, 4834, 26796, 111, 4484, 2097, 105, 21764, 7861, 102, 21900, 102, 60, 21426, 4553, 8331, 8807, 105, 340, 4663, 47, 11, 5564, 2021, 46, 1451, 9104, 275, 6549, 1858, 501, 33, 226, 156, 130, 226, 155, 180, 226, 156, 172, 226, 156, 152, 226, 155, 169, 226, 155, 184, 226, 156, 172, 226, 155, 184, 226, 156, 155, 226, 155, 169, 226, 156, 140, 226, 156, 172, 226, 156, 151, 226, 155, 170, 226, 156, 144, 226, 155, 171, 226, 155, 191, 226, 156, 172, 226, 155, 170, 226, 155, 191, 226, 156, 159, 226, 156, 172, 226, 155, 188, 226, 156, 130, 226, 156, 144, 226, 156, 172, 226, 155, 191, 226, 156, 151, 226, 156, 172, 226, 155, 188, 226, 156, 151, 226, 155, 171, 226, 155, 178, 226, 156, 152, 226, 156, 130, 226, 155, 171, 226, 155, 168, 226, 156, 172, 226, 156, 152, 226, 156, 151, 226, 156, 173, 11, 5564, 2021, 46, 1451, 9104, 275, 33226, 501, 308, 100, 344, 2500, 104, 4556, 9277, 336, 8422, 111, 4712, 101, 21824, 4682, 31063, 8168, 2510, 4660, 47, 11, 40607, 50199, 59, 20108, 30388, 102, 4556, 1751, 4522, 1853, 21265, 21824, 4571, 2169, 106, 2524, 4660, 22188, 201, 158, 117, 47, 11, 48487, 59, 308, 21413, 21592, 38598, 21265, 4601, 38278, 461, 31123, 4660, 47, 11, 48487, 275, 6077, 501, 326, 98, 2632, 342, 9275, 340, 2666, 117, 4556, 2623, 2666, 116, 4967, 2062, 33, 2632, 117, 335, 2622, 123, 345, 2624, 117, 339, 202, 157, 2666, 117, 4663, 2666, 94, 275, 53652, 20673, 2078, 26293, 27168, 42, 11, 48487, 275, 5649, 25794, 501, 33, 227, 161, 139, 10071, 227, 161, 138, 227, 161, 130, 227, 161, 158, 10071, 227, 161, 146, 227, 161, 130, 227, 161, 159, 10071, 227, 161, 156, 227, 161, 136, 227, 161, 130, 227, 161, 143, 227, 161, 143, 10071, 227, 161, 130, 227, 161, 158, 227, 161, 154, 10071, 227, 161, 139, 227, 161, 159, 10071, 227, 161, 154, 227, 161, 150, 227, 161, 146, 227, 161, 143, 227, 161, 158, 227, 161, 159, 10071, 227, 161, 148, 227, 161, 166, 227, 161, 152, 227, 161, 159, 10071, 227, 161, 142, 227, 161, 146, 11, 6134, 6953, 1746, 59, 4027, 4620, 1746, 4686, 1745, 4556, 1751, 21798, 340, 22156, 98, 21834, 4663, 47, 11, 1217, 8101, 2062, 37320, 282, 19767, 1939, 59, 3631, 21413, 21592, 21762, 2194, 45, 4601, 21542, 2059, 102, 31123, 4835, 47, 11, 1065, 1751, 42518, 59, 275, 1270, 969, 969, 42, 11, 5981, 7692, 275, 53, 501, 33, 241, 145, 141, 189, 241, 145, 141, 177, 241, 145, 141, 179, 33, 241, 145, 141, 179, 241, 145, 141, 188, 241, 145, 141, 181, 241, 145, 142, 132, 33, 241, 145, 141, 186, 2676, 241, 145, 142, 133, 241, 145, 141, 177, 241, 145, 141, 190, 45, 33, 241, 145, 141, 190, 241, 145, 141, 186, 33, 241, 145, 141, 189, 241, 145, 141, 186, 241, 145, 142, 132, 33, 241, 145, 142, 134, 241, 145, 141, 192, 33, 241, 145, 141, 190, 241, 145, 141, 180, 241, 145, 141, 177, 241, 145, 141, 190, 33, 241, 145, 141, 178, 241, 145, 142, 131, 241, 145, 141, 186, 241, 145, 141, 179, 241, 145, 141, 179, 241, 145, 141, 186, 241, 145, 141, 185, 47, 11, 6392, 20525, 2180, 275, 6549, 1858, 501, 33, 226, 156, 151, 226, 155, 181, 33, 226, 155, 184, 226, 156, 151, 226, 156, 144, 33, 226, 156, 151, 226, 156, 144, 226, 156, 130, 33, 226, 155, 168, 33, 226, 155, 184, 226, 156, 155, 226, 156, 151, 226, 155, 178, 33, 226, 156, 153, 226, 155, 191, 33, 226, 155, 167, 226, 156, 151, 226, 156, 140, 226, 156, 140, 33, 226, 155, 169, 226, 155, 168, 33, 226, 155, 162, 226, 156, 151, 33, 226, 155, 178, 226, 155, 168, 226, 155, 169, 33, 226, 156, 140, 226, 155, 169, 226, 155, 178, 11, 6392, 20525, 2180, 275, 33226, 501, 3782, 21751, 4522, 106, 2510, 21762, 115, 4962, 111, 4985, 7512, 332, 2510, 22700, 2510, 98, 350, 9248, 47, 11, 6336, 2186, 282, 58920, 275, 1302, 8258, 2186, 501, 3779, 21942, 21642, 4556, 1751, 4836, 1746, 4966, 4784, 6911, 22077, 47, 11, 6336, 2186, 282, 58920, 275, 890, 1995, 9271, 501, 3933, 104, 21942, 4789, 7909, 38598, 4836, 1853, 4966, 4784, 6914, 22077, 47, 11, 71, 9364, 2108, 2186, 117, 282, 19878, 8310, 102, 59, 3779, 31201, 21642, 4556, 1751, 45, 4784, 98, 2510, 24514, 8765, 47, 11, 2475, 2187, 7472, 1984, 282, 50328, 1939, 59, 4945, 104, 21751, 4522, 106, 2510, 21762, 115, 4962, 111, 4985, 7512, 332, 2510, 4660, 106, 2510, 98, 22091, 47, 11, 84, 27413, 1984, 282, 50721, 59, 20139, 21942, 4965, 2202, 4556, 1751, 4836, 1746, 21303, 4784, 6911, 22091, 47, 11, 5791, 2186, 282, 44188, 59, 3933, 104, 21942, 4789, 7909, 4556, 1751, 45, 21525, 338, 9364, 31141, 4712, 1833, 22344, 22091, 47, 11, 84, 2518, 26295, 107, 2337, 108, 59, 33, 2468, 4615, 4967, 102, 38598, 4827, 1853, 4424, 21525, 4559, 344, 2500, 22153, 102, 47, 11, 1036, 2337, 108, 282, 19913, 2184, 1746, 59, 3910, 21955, 4556, 9303, 4601, 102, 45, 4601, 30655, 117, 4660, 22166, 22446, 115, 47, 11, 1294, 7341, 34962, 282, 36599, 59, 3910, 21942, 4556, 1751, 4522, 1853, 45, 21813, 4491, 1859, 344, 197, 180, 30953, 4633, 1733, 101, 47, 11, 6164, 7236, 2516, 6921, 107, 48, 67, 2514, 24991, 8704, 26696, 59, 308, 7405, 4617, 111, 4556, 1733, 116, 4969, 27651, 45, 22081, 274, 117, 21514, 117, 22090, 1791, 4608, 2065, 22288, 2065, 47, 11, 841, 26895, 7014, 59, 3782, 21942, 4556, 1751, 4508, 117, 45, 31349, 21543, 4491, 1853, 4677, 22173, 4784, 6914, 22173, 47, 11, 77, 9310, 2349, 1778, 25313, 8688, 117, 282, 20325, 7452, 8956, 7911, 59, 3777, 105, 21942, 28969, 4587, 2194, 1853, 45, 4477, 1752, 21514, 117, 22096, 22204, 117, 4858, 106, 47, 11, 959, 35948, 282, 44340, 59, 20108, 31201, 28969, 30778, 111, 45, 31572, 22096, 4890, 22442, 24430, 47, 11, 1423, 1930, 1818, 35948, 59, 20108, 31201, 28969, 22700, 8010, 25234, 8871, 7444, 45, 31572, 21507, 8797, 31419, 22749, 4612, 7253, 111, 22636, 47, 11, 23852, 7465, 7438, 1857, 29592, 117, 59, 3917, 1791, 31201, 309, 2007, 1751, 4620, 2046, 7444, 45, 352, 2233, 2063, 21507, 117, 31419, 21507, 117, 4858, 105, 335, 27635, 101, 47, 11, 1217, 8973, 2225, 1857, 29432, 7048, 3415, 1224, 8683, 7745, 5175, 20108, 4626, 7012, 3849, 2099, 116, 21298, 111, 22657, 4491, 1751, 4497, 1817, 22081, 115, 4686, 4861, 106, 47, 11, 1329, 1853, 120, 9263, 7341, 25938, 59, 308, 7405, 21962, 111, 4556, 1733, 8688, 22725, 1791, 24830, 8792, 111, 31572, 30578, 4521, 22096, 4505, 1776, 4477, 2239, 21561, 111, 4497, 101, 47, 11, 84, 36070, 25938, 282, 20860, 2098, 59, 274, 8688, 31201, 3849, 2102, 21640, 111, 45, 31572, 30578, 460, 1791, 22081, 4858, 1932, 2230, 47, 11, 81, 103, 9263, 2351, 8688, 59, 3917, 1791, 21962, 111, 36734, 30914, 2180, 31572, 30578, 1858, 22081, 4505, 7144, 21309, 26193, 102, 4497, 101, 47, 11, 6608, 120, 2498, 7158, 1791, 282, 4223, 6894, 1746, 59, 308, 342, 2499, 28969, 4542, 9267, 2176, 45, 4712, 101, 21524, 31350, 117, 4670, 4686, 121, 34, 11, 959, 35948, 275, 1563, 8562, 24863, 501, 308, 4615, 4556, 1751, 21591, 2176, 45, 31572, 30578, 22070, 4858, 105, 4816, 1752, 47, 11, 5625, 8619, 1791, 282, 19495, 33864, 59, 308, 4626, 105, 3849, 2102, 21635, 45, 22657, 4521, 4497, 7035, 4656, 22158, 4858, 106, 47, 11, 5559, 34286, 25938, 59, 308, 4615, 2239, 3849, 2099, 116, 30778, 111, 45, 4521, 4816, 1752, 4656, 22158, 4858, 105, 47, 11, 6608, 2295, 9157, 101, 9384, 8688, 275, 91, 9382, 7760, 501, 20108, 30431, 3849, 1733, 116, 4965, 2194, 102, 45, 21506, 22442, 6922, 22096, 345, 9350, 47, 11, 6608, 2295, 9157, 101, 9384, 8688, 275, 1224, 9157, 111, 501, 3777, 105, 21426, 3849, 2496, 116, 4965, 2194, 102, 45, 21506, 22442, 6922, 22081, 22158, 47, 11, 1369, 7084, 7351, 2218, 1791, 59, 275, 1270, 969, 969, 42, 11, 23744, 33864, 59, 20404, 32315, 110, 4517, 2067, 4430, 4984, 9014, 1859, 45, 22161, 22000, 123, 351, 9430, 102, 30223, 110, 47, 11, 1466, 8357, 282, 50236, 59, 4309, 1950, 4797, 9350, 2498, 21985, 1937, 45, 4778, 4512, 4840, 25597, 8333, 98, 22095, 2226, 47, 11, 6602, 106, 275, 53604, 501, 318, 2495, 1932, 9245, 21378, 8569, 22051, 2176, 45, 21507, 4591, 4639, 1752, 333, 9251, 2536, 2536, 1751, 47, 11, 999, 9161, 111, 59, 47898, 5119, 2810, 43073, 32779, 27982, 27972, 27909, 45, 5095, 2821, 5098, 2821, 28158, 5117, 2803, 28065, 28152, 2801, 2823, 5091, 48173, 47, 11, 53604, 20183, 34262, 59, 4027, 102, 4848, 1950, 4797, 2270, 2498, 21985, 1941, 4799, 22095, 2007, 4512, 22222, 4620, 8496, 47, 11, 53693, 20183, 34262, 59, 20422, 2498, 4848, 1950, 4797, 2270, 2498, 4793, 40, 2095, 8122, 21500, 22095, 8911, 4512, 22222, 4620, 7152, 47, 11, 1560, 8530, 1746, 59, 275, 1270, 969, 969, 42, 11, 1563, 2210, 1746, 59, 275, 1270, 969, 969, 42, 11, 1219, 9044, 7745, 59, 275, 1270, 969, 969, 42, 11, 5896, 35216, 59, 4021, 22742, 1950, 4623, 1733, 2184, 22554, 2522, 98, 45, 22449, 4512, 22578, 22118, 2011, 22089, 6938, 47, 11, 6194, 9024, 111, 59, 3789, 22692, 118, 33, 2542, 2195, 4793, 1947, 2026, 45, 22569, 22068, 4682, 1984, 1956, 2542, 47, 11, 6214, 7724, 8241, 111, 59, 300, 2587, 21729, 1957, 22689, 1912, 2210, 4793, 1947, 8138, 4599, 4608, 116, 22068, 2546, 116, 4682, 9445, 1940, 2605, 1937, 11, 6392, 4135, 42473, 59, 275, 1270, 969, 969, 42, 11, 1461, 2152, 7745, 275, 1589, 2062, 7911, 501, 275, 1270, 969, 969, 42, 11, 68, 27548, 59, 20432, 118, 341, 27695, 4784, 2021, 45, 22167, 1770, 2507, 27785, 4663, 47, 11, 1458, 8443, 108, 59, 312, 2514, 9445, 110, 4607, 116, 2591, 4784, 2021, 47, 4063, 2357, 1746, 2507, 4656, 47, 11, 6462, 8724, 282, 44720, 59, 20431, 2546, 4607, 27776, 4798, 108, 9425, 340, 4663, 22173, 4798, 1997, 1839, 106, 47, 11, 1458, 26618, 7745, 59, 20256, 1997, 4607, 110, 22520, 1994, 112, 45, 4682, 4477, 4439, 4663, 5005, 1997, 1828, 9006, 112, 47, 11, 890, 2189, 7745, 45, 50135, 111, 45, 37042, 8224, 8615, 21265, 50661, 275, 33226, 501, 3932, 22102, 118, 31185, 106, 22518, 1994, 112, 45, 340, 4811, 4663, 4682, 5005, 8797, 106, 47, 11, 890, 2189, 7745, 45, 37042, 8224, 8615, 21265, 50661, 275, 931, 8613, 8087, 501, 5061, 2794, 57766, 32823, 43109, 48146, 42832, 45, 5099, 32786, 32732, 32740, 5115, 43136, 2802, 47, 11, 6259, 7415, 8241, 111, 59, 32609, 27933, 2806, 32691, 5123, 27865, 27871, 48146, 42832, 45, 5091, 32740, 32731, 5115, 54741, 47, 11, 48701, 59, 5090, 57766, 57704, 48147, 42832, 45, 32748, 2808, 5103, 28007, 32740, 47971, 27908, 2812, 47, 11, 5628, 24694, 7745, 275, 931, 8613, 8087, 501, 5090, 32730, 27903, 5096, 2811, 28116, 5115, 42832, 45, 5119, 28009, 5103, 28007, 32740, 48194, 2798, 27949, 28113, 47, 11, 5628, 24694, 7745, 275, 1217, 7247, 1984, 501, 3932, 22062, 118, 4607, 27773, 5005, 108, 9425, 45, 21911, 112, 4667, 102, 4682, 5005, 1997, 1839, 7956, 47, 11, 1532, 26796, 7745, 59, 5090, 48053, 2813, 5122, 43109, 32773, 27986, 45, 5121, 32679, 28009, 48050, 2830, 32740, 32703, 43233, 27908, 28079, 47, 11, 5655, 7604, 7745, 59, 32609, 27899, 32691, 5119, 2806, 32778, 2820, 42832, 45, 32786, 32740, 32732, 47971, 27908, 47, 11, 33132, 7745, 59, 33, 9690, 9687, 9691, 9679, 9695, 33, 9684, 2998, 174, 9679, 9690, 33, 9682, 9679, 33, 9679, 9694, 9679, 33, 9690, 9696, 9688, 9687, 9684, 9679, 47, 11, 5577, 25272, 111, 59, 33, 213, 192, 2878, 2871, 2857, 2869, 33, 2857, 2874, 2857, 2866, 2864, 33, 2873, 2880, 2877, 2861, 2865, 33, 215, 136, 33, 2864, 2871, 214, 175, 2864, 33, 2857, 2871, 2867, 2857, 2871, 2859, 2864, 2875, 2877, 33, 214, 186, 214, 169, 2871, 2861, 2878, 215, 138, 11, 847, 7123, 7745, 59, 4286, 2505, 31470, 22645, 4567, 348, 1851, 114, 4484, 102, 4697, 108, 344, 2505, 4555, 1853, 4555, 2505, 47, 11, 6719, 1991, 2183, 59, 19616, 4880, 2327, 7154, 7897, 110, 45, 21330, 98, 22805, 24699, 21553, 2239, 8155, 47, 11, 6719, 1991, 2183, 275, 1340, 8827, 1746, 501, 5141, 28212, 5158, 2954, 2951, 5139, 2948, 28236, 2949, 5139, 219, 174, 2924, 33, 2939, 2934, 2934, 2953, 33, 2940, 2952, 2946, 28239, 2949, 2935, 11, 1494, 8780, 59, 32559, 27871, 32691, 32668, 2810, 45, 5106, 2821, 2825, 27980, 45, 33, 2851, 2806, 27994, 32672, 48110, 27964, 32732, 28007, 47, 11, 86, 2346, 1850, 282, 314, 9820, 2346, 1850, 7229, 59, 275, 33397, 501, 20409, 4781, 25946, 4879, 2329, 8712, 110, 22119, 8025, 45, 4418, 2048, 352, 22080, 1891, 22805, 1750, 21949, 2210, 2162, 1757, 1822, 47, 11, 86, 2346, 1850, 282, 33, 209, 143, 2801, 27883, 2804, 28117, 275, 931, 8613, 8087, 501, 32607, 2807, 32812, 28125, 5096, 2803, 2802, 43228, 32735, 2806, 42828, 45, 5091, 2806, 27998, 5110, 48050, 27899, 32703, 28043, 2810, 5101, 27924, 28074, 2810, 42899, 27908, 47, 11, 23423, 2007, 282, 44021, 106, 59, 33, 2975, 135, 9505, 9514, 22887, 9513, 2975, 130, 2975, 155, 33, 2975, 151, 9517, 9500, 9517, 33, 9503, 9513, 9507, 9514, 45, 33, 9500, 9513, 9500, 9517, 33, 2975, 135, 9505, 9513, 9507, 22887, 9518, 9502, 9518, 22887, 9519, 2975, 184, 9500, 9514, 33, 9511, 2976, 160, 33, 9502, 9513, 9496, 11, 6266, 24748, 275, 8151, 7302, 7854, 501, 22881, 9489, 22874, 9487, 9464, 33, 9462, 9487, 2973, 139, 33, 9482, 9461, 9469, 9494, 45, 22881, 9480, 9487, 22876, 9492, 22877, 9490, 9462, 9469, 22878, 9487, 9485, 9489, 47, 11, 6266, 24748, 275, 7551, 25867, 102, 501, 3340, 9477, 9489, 22874, 9487, 9464, 33, 9462, 9487, 2973, 139, 33, 9482, 9461, 9469, 9492, 45, 22881, 9480, 9487, 22876, 9492, 22877, 9490, 9462, 9469, 22878, 9487, 9485, 9489, 47, 11, 1187, 2072, 6911, 59, 33, 2986, 169, 2986, 169, 2986, 152, 2987, 135, 33, 2986, 186, 2986, 191, 2986, 169, 2986, 192, 33, 2986, 135, 2986, 152, 2986, 167, 2987, 135, 45, 33, 2986, 169, 2986, 191, 2986, 169, 2987, 130, 33, 2986, 152, 2986, 157, 2986, 169, 9559, 2986, 169, 2987, 130, 33, 2986, 165, 2986, 192, 2986, 169, 2986, 173, 2986, 186, 2987, 130, 2986, 167, 2987, 130, 11, 1097, 2062, 106, 275, 8151, 7302, 7854, 501, 22881, 9493, 9458, 22874, 9487, 2973, 130, 9464, 33, 9462, 9487, 22885, 9461, 9469, 9487, 22886, 9491, 2973, 130, 33, 2973, 149, 9479, 22881, 9490, 2973, 158, 9492, 33, 2973, 138, 9484, 9484, 9492, 22874, 9494, 2973, 137, 33, 9464, 9494, 9466, 22878, 9485, 9489, 9458, 22879, 9485, 9490, 9458, 9464, 9469, 9489, 47, 11, 1097, 2062, 106, 275, 7551, 25867, 102, 501, 3340, 9477, 9493, 9458, 22874, 9487, 2973, 130, 9464, 33, 9462, 9487, 22885, 9461, 9469, 9489, 22886, 9491, 2973, 130, 33, 2973, 149, 9479, 22881, 9490, 2973, 158, 9492, 33, 2973, 138, 9484, 9484, 9492, 22874, 9494, 2973, 137, 33, 9464, 9494, 9466, 22878, 9485, 9489, 9458, 22879, 9485, 9490, 9458, 9464, 9469, 9489, 47, 11, 6263, 7100, 8055, 59, 33, 9562, 9571, 9584, 9563, 9588, 9563, 9588, 33, 2988, 152, 9588, 9577, 9583, 9582, 9588, 33, 9569, 9584, 9571, 9588, 9571, 9583, 9560, 47, 33, 9561, 9569, 9586, 9571, 9588, 9571, 9586, 33, 9579, 9587, 9570, 9571, 9584, 9572, 9588, 9572, 9584, 9563, 9588, 9563, 9584, 9577, 9588, 9577, 47, 11, 6684, 1948, 59, 33, 9538, 9548, 9539, 9552, 33, 9534, 2982, 164, 9552, 2982, 164, 9548, 9536, 9549, 33, 9535, 9548, 9540, 9552, 9540, 9549, 9536, 9550, 9547, 2983, 136, 9539, 9552, 45, 33, 2982, 134, 9537, 9539, 9548, 9545, 9552, 33, 2982, 143, 9539, 9534, 9552, 9534, 9550, 33, 2982, 147, 9543, 9550, 33, 9534, 2983, 136, 9536, 9550, 9541, 9552, 33, 9547, 9543, 9548, 9537, 9550, 47, 11, 6693, 8895, 59, 33, 2984, 169, 2985, 136, 2984, 169, 9556, 33, 2984, 152, 9554, 2984, 157, 9556, 33, 2984, 165, 9555, 2984, 169, 2984, 152, 2984, 179, 2984, 169, 9556, 33, 2984, 175, 9553, 9555, 2984, 176, 9556, 33, 2984, 134, 2984, 179, 9554, 33, 2984, 155, 2985, 136, 2984, 185, 9555, 2984, 169, 9554, 33, 2984, 169, 9554, 2984, 150, 9556, 33, 2984, 144, 2984, 175, 9555, 33, 2984, 136, 2984, 173, 9557, 2984, 173, 2984, 131, 2984, 167, 9555, 33, 2984, 179, 2985, 136, 2984, 167, 9556, 11, 6629, 7663, 7506, 59, 33, 9595, 2990, 168, 33, 9597, 2991, 148, 2990, 176, 2991, 149, 9596, 2991, 149, 33, 9593, 2991, 146, 9595, 2990, 168, 33, 2991, 133, 2991, 145, 9593, 9600, 2990, 187, 9600, 47, 33, 2990, 146, 2990, 187, 9600, 9594, 9598, 33, 9595, 2990, 168, 33, 9593, 9600, 2991, 132, 9600, 33, 2991, 133, 9599, 9594, 9600, 2990, 187, 9593, 9598, 33, 2991, 132, 9600, 2990, 176, 2991, 149, 33, 9594, 2991, 157, 9597, 2991, 155, 47, 11, 1537, 1834, 41, 52, 501, 32840, 219, 187, 5160, 28213, 2962, 5160, 2965, 2924, 5146, 2963, 2927, 2924, 33, 2966, 2952, 219, 187, 5138, 28236, 5154, 2929, 2965, 2968, 5140, 2963, 2948, 2967, 2945, 5155, 2966, 2967, 219, 187, 33, 2966, 2952, 2927, 2967, 33, 220, 149, 11, 6448, 1932, 112, 41, 52, 501, 33, 2935, 2951, 5147, 2954, 2937, 2951, 5143, 2952, 219, 148, 2948, 220, 145, 5147, 2949, 2918, 5156, 2943, 2951, 5154, 2924, 5155, 2951, 5143, 2952, 219, 151, 2952, 2954, 11, 5919, 2184, 282, 50538, 41, 52, 501, 281, 2949, 2950, 32840, 5140, 2952, 28213, 2949, 5139, 2932, 28239, 2958, 5138, 2930, 2936, 28207, 32835, 2932, 5147, 2954, 2937, 2951, 5139, 2931, 28236, 2949, 11, 23399, 1939, 41, 52, 501, 32826, 2924, 5151, 28204, 2934, 32837, 2953, 5136, 2947, 2948, 32829, 2935, 2929, 2924, 2929, 5157, 5156, 2933, 2924, 5153, 2924, 5158, 217, 165, 2948, 2949, 2950, 2954, 47, 11, 852, 6991, 1939, 59, 275, 1270, 969, 969, 42, 11, 6263, 8796, 102, 59, 4065, 8749, 40, 22173, 8031, 4594, 46, 2554, 2552, 1941, 2552, 352, 4656, 21907, 2554, 26246, 106, 4871, 1974, 47, 11, 1095, 24924, 41, 52, 501, 5124, 2907, 2900, 5128, 2902, 2896, 2903, 5130, 2891, 2902, 2896, 2903, 33, 2897, 2902, 2896, 2902, 28198, 5127, 2897, 2895, 5130, 2891, 5131, 2897, 2900, 2914, 5130, 2900, 47, 11, 90, 7770, 7911, 41, 52, 501, 5124, 2900, 2901, 33, 2914, 2909, 2906, 5133, 2908, 2906, 33, 2893, 2903, 2891, 2887, 2897, 5124, 2896, 2906, 5133, 2908, 33, 2899, 2896, 2899, 5131, 2900, 2915, 5132, 2900, 2916, 2899, 33, 216, 177, 216, 179, 47, 11, 6148, 1854, 46, 23399, 1939, 59, 275, 1270, 969, 969, 42, 11, 1217, 7356, 112, 59, 275, 1270, 969, 969, 42, 11, 72, 200, 158, 2656, 200, 158, 123, 59, 275, 1270, 969, 969, 42, 11, 848, 7668, 1939, 59, 275, 1270, 969, 969, 42, 11, 1505, 106, 59, 20412, 8925, 21316, 22632, 8467, 45, 33, 202, 157, 2082, 202, 157, 4660, 4580, 1844, 47, 11, 1094, 8968, 275, 33226, 501, 3913, 98, 2675, 4604, 98, 22571, 8213, 4554, 2007, 2675, 8712, 4631, 2034, 4596, 21730, 98, 2675, 4635, 2675, 1874, 2323, 2675, 47, 11, 1094, 8968, 275, 845, 6995, 42, 275, 51, 501, 5137, 2958, 28231, 5137, 2958, 2953, 2956, 5140, 2956, 28239, 2956, 2934, 33, 2943, 2958, 2948, 2956, 28208, 2958, 5152, 2957, 2949, 2956, 5137, 2958, 2950, 33, 2943, 2956, 2949, 2956, 2924, 5153, 2956, 28210, 2958, 2953, 2956, 2924, 11, 1629, 8656, 98, 41, 53, 501, 4030, 343, 2502, 4607, 2681, 21587, 104, 2507, 45, 342, 2512, 345, 2507, 4725, 22131, 22051, 2151, 47, 11, 6212, 7600, 98, 59, 20491, 8336, 2672, 21960, 106, 2672, 2323, 30292, 2672, 2067, 4450, 98, 22093, 8937, 106, 45, 4514, 8413, 2672, 2007, 4684, 98, 2672, 106, 2672, 4656, 1763, 2672, 351, 2628, 2672, 47, 11, 41, 1190, 42, 1467, 6947, 2015, 59, 4060, 2278, 2345, 4631, 2007, 21359, 7082, 106, 4678, 4782, 8041, 2339, 106, 47, 11, 6263, 1757, 59, 20861, 98, 21374, 1847, 22065, 1746, 4615, 1784, 21503, 4583, 22597, 1743, 22080, 7219, 7487, 8007, 22441, 98, 47, 11, 6680, 24526, 59, 20187, 98, 21962, 104, 4631, 26199, 22148, 104, 21391, 2091, 4424, 31086, 106, 4416, 112, 22071, 6973, 1746, 47, 11, 5726, 8186, 2164, 59, 4212, 9329, 4881, 40, 4459, 8929, 7235, 21967, 27069, 45, 4806, 4567, 4678, 40, 2007, 8055, 1853, 4881, 446, 11, 1032, 26051, 59, 3643, 22363, 98, 4686, 21942, 98, 4594, 8329, 112, 45, 4583, 4425, 22450, 98, 4686, 4840, 1984, 7203, 7009, 21955, 98, 47, 11, 23808, 8234, 102, 59, 3634, 118, 4600, 98, 38967, 111, 21344, 7856, 22566, 2111, 21984, 98, 47, 11, 5656, 8164, 102, 275, 48771, 287, 47, 49, 501, 33, 9664, 2997, 186, 2997, 155, 2997, 186, 2997, 158, 9667, 2997, 186, 9809, 9666, 9675, 9672, 2997, 186, 9809, 226, 130, 139, 9664, 2997, 186, 2997, 155, 2997, 186, 2997, 158, 9667, 2997, 186, 9809, 9669, 33, 9669, 2997, 186, 2997, 155, 9664, 2997, 186, 9809, 2997, 134, 9672, 9676, 9667, 9674, 9673, 9665, 2997, 186, 9809, 9671, 2997, 139, 2997, 186, 9809, 226, 130, 140, 33, 226, 130, 143, 9664, 2997, 186, 9670, 9675, 9672, 9665, 2997, 186, 9809, 2997, 184, 33, 2997, 146, 9673, 2997, 130, 9674, 9673, 9664, 2997, 186, 9809, 9669, 2997, 186, 2997, 160, 9674, 33, 9669, 9670, 2997, 186, 2997, 160, 9673, 9668, 9672, 226, 130, 140, 275, 58, 42, 11, 5656, 8164, 102, 275, 48771, 288, 47, 49, 501, 33, 9664, 2997, 188, 2997, 190, 9667, 9677, 9666, 9675, 9672, 9677, 33, 9664, 2997, 188, 2997, 190, 9667, 9677, 9669, 33, 9669, 2997, 191, 9667, 9677, 2997, 134, 9672, 9676, 9667, 9673, 9674, 9665, 9677, 9666, 2997, 155, 9677, 226, 130, 140, 33, 226, 130, 143, 9665, 9677, 9676, 9664, 9678, 9675, 9672, 9665, 9677, 2997, 184, 33, 2997, 146, 9673, 2997, 130, 9673, 9674, 9664, 9677, 9669, 2997, 191, 9674, 9669, 9670, 2997, 191, 9673, 9668, 2997, 172, 226, 130, 140, 275, 58, 42, 11, 24296, 41878, 275, 2148, 28285, 4684, 9768, 501, 319, 9346, 21492, 39952, 33, 9396, 4805, 9764, 122, 22601, 105, 22126, 46364, 339, 28250, 338, 2506, 47, 11, 24296, 41878, 275, 111, 9347, 42, 275, 53, 501, 33, 10349, 33, 241, 164, 143, 144, 33, 10267, 33, 3051, 186, 33, 13676, 33, 13130, 33, 241, 167, 148, 162, 33, 15090, 33, 241, 164, 143, 144, 33, 11917, 33, 11141, 11, 1189, 8163, 59, 33, 2999, 130, 9709, 2999, 138, 2999, 188, 3000, 135, 2999, 163, 9708, 2999, 134, 2999, 138, 2999, 188, 3000, 135, 9704, 2999, 138, 9709, 2999, 134, 9704, 3000, 140, 2999, 149, 9708, 9705, 33, 2999, 139, 3000, 133, 2999, 154, 2999, 131, 9709, 2999, 153, 9708, 9705, 2999, 149, 2999, 138, 9709, 2999, 161, 9708, 9706, 11, 1217, 112, 59, 33, 2994, 131, 2994, 174, 225, 188, 138, 2994, 142, 2994, 130, 2994, 181, 2994, 154, 225, 188, 130, 2994, 130, 225, 188, 138, 2994, 168, 225, 188, 133, 2994, 149, 225, 188, 138, 225, 188, 131, 2994, 149, 2994, 142, 2994, 152, 2994, 182, 225, 188, 137, 2994, 162, 2994, 178, 2994, 154, 2994, 155, 225, 188, 142, 225, 188, 137, 225, 188, 133, 2994, 149, 225, 188, 138, 225, 188, 129, 2994, 175, 2994, 178, 2994, 149, 225, 188, 132, 2994, 172, 225, 188, 138, 2994, 131, 2994, 174, 225, 188, 138, 2994, 142, 225, 188, 129, 2994, 137, 2994, 178, 2994, 155, 47, 11, 1497, 1741, 59, 33, 2992, 138, 9631, 9614, 9601, 9634, 9614, 9601, 43262, 9605, 9601, 9643, 9609, 9646, 33, 9640, 9610, 9645, 9620, 9631, 9614, 43268, 9645, 9612, 9633, 9642, 9628, 9646, 2992, 138, 9631, 9614, 9639, 9605, 9644, 9615, 11, 33280, 8085, 111, 275, 931, 8613, 8087, 501, 32564, 32812, 2805, 5099, 2798, 28151, 32805, 42737, 45, 48059, 27865, 32800, 28048, 28023, 32670, 2818, 11, 33280, 8085, 111, 275, 48410, 42, 275, 54, 501, 33, 226, 161, 171, 226, 161, 163, 33, 226, 161, 177, 226, 161, 163, 226, 161, 176, 226, 161, 163, 33, 226, 161, 163, 226, 161, 180, 226, 161, 162, 226, 161, 183, 226, 161, 167, 33, 226, 161, 181, 226, 161, 163, 226, 161, 180, 226, 161, 161, 226, 161, 169, 226, 161, 161, 33, 226, 161, 131, 33, 226, 161, 169, 226, 161, 161, 226, 161, 180, 226, 161, 165, 226, 161, 184, 33, 226, 161, 173, 226, 161, 164, 226, 161, 165, 226, 161, 184, 226, 161, 161, 226, 161, 180, 226, 161, 161, 226, 161, 163, 33, 226, 161, 171, 226, 161, 163, 226, 161, 177, 226, 161, 163, 11, 69, 9167, 104, 1990, 98, 59, 275, 1270, 969, 969, 42, 11, 1294, 8466, 106, 59, 23243, 9477, 22874, 9487, 2973, 130, 9464, 33, 9462, 9487, 9473, 22885, 9461, 9495, 2973, 156, 9491, 22882, 22881, 9480, 9487, 2973, 137, 22874, 9492, 9485, 9488, 22878, 9489, 22886, 9490, 9473, 9495, 9810, 9473, 9495, 33, 9496, 11, 1498, 24869, 111, 59, 33, 2996, 165, 9661, 9657, 9649, 9658, 225, 191, 147, 9662, 9649, 2996, 160, 9649, 9653, 9658, 9649, 9651, 9649, 9653, 9649, 9650, 9659, 9649, 9655, 9649, 9656, 9661, 9652, 2995, 142, 11, 48407, 59, 33, 12605, 15752, 11056, 10258, 14452, 14543, 15646, 10260, 10429, 17038, 10452, 28329, 48407, 275, 24254, 49331, 501, 33, 12605, 15752, 11056, 10258, 14452, 14543, 15646, 10260, 10616, 17038, 18215, 28329, 6681, 9057, 7506, 41, 55, 501, 19998, 98, 33, 2542, 46, 117, 27587, 21428, 98, 2678, 105, 4738, 46, 109, 2504, 45, 344, 2527, 333, 2542, 4806, 112, 2678, 105, 46, 2184, 8367, 47, 11, 53571, 59, 33, 15028, 10139, 10176, 10233, 10189, 10164, 18053, 10146, 43435, 43421, 10080, 43356, 10139, 15028, 10164, 10616, 10129, 10113, 58572, 28329, 1192, 8576, 111, 59, 23006, 18585, 23142, 43656, 33, 18694, 18884, 23100, 23150, 18830, 18862, 47, 23001, 18641, 18606, 23108, 19045, 18934, 23110, 43731, 11, 888, 2187, 6991, 59, 4027, 31946, 4615, 1984, 102, 4556, 1751, 45, 21808, 106, 4692, 31946, 21946, 1852, 4663, 47, 11, 6000, 1741, 7745, 59, 3880, 1991, 4583, 2655, 118, 4617, 33, 2655, 1741, 340, 4617, 4419, 25641, 60, 33, 2655, 98, 2655, 8343, 345, 2573, 343, 2527, 4425, 336, 33, 2655, 1847, 98, 47, 11, 6266, 26773, 1746, 59, 304, 4626, 2655, 7003, 336, 4615, 106, 340, 4803, 21944, 6947, 45, 4660, 98, 33, 2655, 2527, 45, 33, 2655, 98, 2655, 102, 4567, 2233, 1753, 47, 11, 1136, 2236, 8818, 2244, 275, 630, 501, 33, 226, 145, 139, 226, 148, 150, 226, 147, 142, 226, 151, 134, 33, 226, 148, 131, 226, 150, 135, 226, 149, 174, 226, 150, 141, 226, 151, 148, 226, 147, 132, 226, 146, 176, 33, 226, 148, 178, 226, 150, 140, 226, 154, 178, 226, 146, 167, 226, 146, 145, 226, 148, 145, 226, 148, 136, 226, 151, 134, 226, 146, 145, 226, 151, 148, 11, 5728, 2073, 2095, 20145, 7641, 59, 4060, 7823, 344, 2627, 1995, 2627, 108, 4615, 2000, 105, 2627, 117, 21976, 8982, 122, 45, 4733, 4858, 2330, 4829, 2236, 22120, 108, 46, 2184, 108, 4678, 122, 47, 11, 6326, 6964, 59, 4257, 9298, 2617, 2656, 4880, 2183, 2530, 2672, 9401, 112, 333, 2507, 9319, 9323, 1920, 1914, 105, 21588, 2513, 4491, 112, 4781, 7959, 4682, 1865, 1891, 106, 4477, 47, 11, 5727, 26928, 102, 275, 7005, 19663, 102, 45, 36468, 7051, 120, 45, 19663, 102, 45, 20416, 8141, 45, 314, 1969, 1780, 98, 45, 20257, 8422, 45, 313, 2495, 7724, 2212, 45, 20682, 7239, 98, 45, 3647, 8150, 98, 45, 21265, 39193, 54830, 59946, 501, 275, 1270, 969, 969, 42, 11, 5969, 1942, 8933, 59, 275, 1270, 969, 969, 42, 11, 1069, 8101, 105, 59, 275, 1270, 969, 969, 42, 11, 1222, 1964, 1746, 59, 4663, 4615, 1996, 102, 4639, 4697, 21438, 1984, 4639, 21367, 1792, 281, 7831, 40, 106, 4639, 4778, 4559, 40, 106, 4678, 355, 8564, 106, 4663, 11, 79, 9342, 7349, 104, 59, 311, 107, 2578, 115, 4879, 30414, 8564, 30534, 2063, 2524, 33, 2523, 4610, 115, 334, 3003, 132, 8564, 47, 11]

def test():
    tokens = tokenizer_encode(test_string)
    assert tokens == expected_tokens, f"\nActual: {tokens}\nExpected: {expected_tokens}"

    decoded_string = tokenizer.decode(tokens)
    assert test_string == decoded_string, f"\nDecoding mismatch:\n{decoded_string}"

    print('All tests pass')

if __name__ == "__main__":
    test()

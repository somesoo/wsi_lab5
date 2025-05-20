Zadanie 5.

Niech dana będzie funkcja f∷[-10,10] -> R o następującej postaci:

f(x) = x^{2} * sin(x) + 100 * sin(x) * cos(x)

Zaimplementuj perceptron wielowarstwowy, który posłuży do aproksymacji funkcji.

Zbadaj:

    wpływ liczby neuronów w warstwie na jakość uzyskanej aproksymacji
    róznice w jakości aproksymacji w zależności od zastosowania metody gradientowej lub metody ewolucyjnej do znajdowania wag sieci. 

Do implementacji zadanego aproksymatora lub solwera optymalizacyjnego możesz korzystać z zewnętrznych bibliotek: np. numpy, scipy, scikit-learn. 




Zadanie do wykonania w parach
Wskazówki od prowadzącego:

Nie można używać torch'a

Można użyć prymitywów do algebry liniowej, do gradientów
Można używać mainstrimowych funkcji aktywacji

Ale implementacja architektury, warstw trzeba implementować sami
Samodzielnie zaimplementować sieć neuronową - to co implementuje neuron, warstwe, mechanizm propagacji w przód i wstecznej
Można użyć gotowy algorytm optymalizacyjny (Adam / svd)
Samo mięso ma być napisane przez nas.



Mamy sobie funkcje f(x) x::R -> R
Znaleźć coś co jest aproksymatorem tej funkcji
Aproksymator ma za zadanie aby w jak najlepszym stopniu naśladowac funkcje f(x)
L(f,f') i to mamy zminimalizować (błąd średnio kwadratowy)(coś takiego)

Zaimplementować w sensowny sposób sieć neuronową (wejściowe, ukryte, wyjściowe)

Badania: - eksperymenty
        - mam sieć z ustaloną liczbą neuronów i warstw
Co się dzieje z aproksymatorem jak zwiększamy ilość neuronów ukrytych
Co się dzieje z aproksymatorem jak zwiększamy ilość warstw ukrytych
        - sprawdzam jak zachowuje się błąd średniokwadratowy


dodatkowo należy:
Rozwiązać to zadanie używając modelu bazowego
bo jest prosty bo jest tani jako model i działa nie gorzej niż sieć neuronowa
(regresja wieloliniowa?) - modelu tego nie trzeba uczyć i powinno działać, parametry analitycznie się wyznacza


należy zwrócić uwagę na czas wykonywania - jak długo to sie będzie uczyć
ma być wykres funkcji aproksymowanej i aproksymatorów w dziedzinie i zobaczyć co się będzie działo przy końcach dziedziny


wniosek: metody globalne wymagają dużo więcej czasu i zasobów
modele takie jak adam czyli lokalne nie są dużo gorsze.
zacznijmy od modelu bazowego typu adam

raczej sigmoidalne




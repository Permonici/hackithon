from __future__ import annotations


SEED_QA: list[dict[str, str]] = [
    {
        "id": "seed-erecept-epoukaz-uhrada",
        "question": "Nejde odeslat ePoukaz a system hlasi problem s uhradou. Co mam zkontrolovat?",
        "answer": "Zkontrolujte vyplnenou diagnozu, typ uhrady a pojistovnu. Pokud se ePoukaz stale neodesle, prilozte presnou hlasku a predejte pripad podpore.",
        "topic": "erecept",
    },
    {
        "id": "seed-erecept-sukl",
        "question": "Nejde odeslat eRecept do SUKL. Co mam udelat?",
        "answer": "Overte pripojeni, platnost certifikatu a prihlaseni uzivatele pro eRecept. Pri opakovane chybe poslete podpore presnou hlasku a cas pokusu.",
        "topic": "erecept",
    },
    {
        "id": "seed-cert-login",
        "question": "Po instalaci certifikatu se uzivatel nemuze prihlasit do XDENTu.",
        "answer": "Overte, ze je certifikat nainstalovany ve spravnem ulozisti a patri spravnemu uzivateli. Pokud prihlaseni stale pada, predejte podpore screenshot a informaci, na kterem pocitaci se chyba deje.",
        "topic": "certificate-authentication-setup",
    },
    {
        "id": "seed-print-template",
        "question": "Dokument se netiskne spravne a potrebuji upravit sablonu tisku.",
        "answer": "Zkontrolujte zvolenou tiskarnu, format dokumentu a prirazenou sablonu. Pokud se tisk lisi od ocekavani, poslete podpore nazev sablony a ukazku vystupu.",
        "topic": "printing-templates-documents",
    },
    {
        "id": "seed-install-update",
        "question": "Po aktualizaci nebo instalaci XDENT nefunguje spravne.",
        "answer": "Nejdriv overte, zda aktualizace probehla na serveru i stanici a zda je dostupne pripojeni k databazi. Pri chybe poslete verzi XDENTu, stanici a presnou hlasku.",
        "topic": "installation-setup",
    },
    {
        "id": "seed-calendar-booking",
        "question": "Kde nastavim objednavani nebo kalendar v XDENTu?",
        "answer": "Hledejte nastaveni kalendare a ordinacnich casu v casti pro objednavani. Pokud neni jasne, ktere nastaveni chybi, poslete podpore popis pozadovane zmeny.",
        "topic": "calendar-scheduling-booking",
    },
    {
        "id": "seed-vzp-batch",
        "question": "VZP nebo pojistovna odmita vykazani. Co zkontrolovat?",
        "answer": "Zkontrolujte kod vykonu, diagnozu, pojistovnu pacienta a pravidla vykazani. Pri zamitnuti pridejte do eskalace chybovou hlasku a konkretni davku.",
        "topic": "vzp",
    },
    {
        "id": "seed-navigation-template",
        "question": "Kde v XDENTu najdu nastaveni sablon dokumentu?",
        "answer": "Hledejte cast pro dokumenty, tisk nebo sablony podle typu vystupu. Pokud si nejste jisti, uvedte nazev dokumentu, ktery chcete upravit.",
        "topic": "how-to-product-navigation",
    },
]

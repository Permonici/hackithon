# Demo scénář

## Hlavní věta

„Ukazujeme AI operátora 1. úrovně podpory, který neodpovídá z hlavy. Nejdřív najde podobné hovory v transkripcích, zkontroluje jistotu a teprve potom odpoví.“

## Co ukázat

1. Status nahoře:
   - OpenAI ready,
   - počet indexovaných chunků,
   - pokrytí témat.

2. Dotaz na ePoukaz:
   - agent rozpozná eRecept/ePoukaz,
   - najde zdroje,
   - odpoví stručně.

3. Evidence cards:
   - každý zdroj má skóre,
   - úryvek z transkripce,
   - možné řešení.

4. Strict mode:
   - pokud je jistota slabá, agent neimprovizuje.

5. Eskalace:
   - při nejistotě připraví balíček pro 2. úroveň podpory.

## Silné stránky pro porotu

- Oddělený frontend a backend.
- Docker Compose spuštění.
- Qdrant jako produkčnější vektorová databáze.
- Hybridní retrieval, ne jen obyčejné hledání slov.
- OpenAI embeddings + chat model.
- Agent timeline vysvětluje chování systému.
- Log interakcí umožňuje zpětné vyhodnocení kvality.

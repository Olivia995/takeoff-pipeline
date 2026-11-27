# GenieMat takeoff pipeline

Turns an architectural floor plan PDF into a Pliteq-style takeoff sheet and a bill of materials.

```
plan.pdf ──detect──▶ rooms.json + rooms_preview.pdf ──select / review──▶ rooms.json ──build──▶ takeoff.pdf + takeoff.csv
```

Everything runs on your own computer. Nothing is sent anywhere.

## The app (recommended)

Double-click **`Start Takeoff App.command`** in this folder. The first time, it installs what it needs (about a minute). Then your browser opens the app at `http://127.0.0.1:8765`.

If macOS blocks it the first time ("unidentified developer"), right-click the file, choose **Open**, then **Open** again.

Or, from Terminal:

```bash
cd takeoff-pipeline
source .venv/bin/activate
python3 -m takeoff serve
```

Keep the Terminal window open while you use the app. Close it, or press Ctrl+C, to stop.

In the app:

1. **Open plan** (or drag a PDF onto the page). The plan is copied into `plans/`, the scale is read from the sheet, and rooms are detected and named automatically. Pick an earlier plan from **Recent plans**.
   **AI first pass with a Claude key: click the rooms.** When a Claude key is saved, **AI first pass** asks you to click inside each room you want (numbered purple dots), then press **Enter**: Claude traces just those rooms, all at once, and adds them ticked. Backspace removes the last click, Esc cancels, and **W** reads the whole sheet instead (below). Nothing is spent on rooms you don't want, and the click puts Claude in the right room. Each room costs about 4¢ on Opus 5.5 or about 1¢ on Sonnet 5.5 (AI settings → Border tracing). In one test on Maroon Creek, Sonnet traced Gym 1 as well as Opus (94%) but stopped Gym 2 at the revision cloud (81% vs 94%), so Opus is the default. The route is `api_ai_trace` in `takeoff/server.py`; the line-traced room under each click frames Claude's picture and gives its name.
   **AI first pass (optional).** Click **AI first pass** instead of Detect rooms. The first time, it asks for an API key (AI settings): a **Claude** key (starts `sk-ant-`, from platform.claude.com → Settings → API keys) or a **Gemini** key. Paste either into the same box; the app tells them apart, checks the key, and saves it on this computer only, in `config/anthropic_key.txt` or `config/gemini_key.txt` (or set `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`). Never paste a key into a chat or email, and leave both files out when sharing the folder. Use **paid** keys: Google's free tier may use content to improve its products.
   **Choosing the model.** AI settings lists the Claude models (Haiku 5.5, Sonnet 5.5, Opus 5.5) when a Claude key is saved, and the Gemini models your Gemini key can use, each with its price and a rough cost per sheet. Model ids starting `claude-` go to Anthropic, the rest to Google. A Claude section is about 6,000 tokens in (it reads the 2048-pixel section at higher resolution) and is asked to think briefly (effort `low`): about 2¢ a sheet on Haiku, 30¢ on Sonnet, 60¢ on Opus. Requests per minute can stay at 300: the Start tier allows 4,000 requests, 2M input and 400K output tokens a minute per model, and a sheet uses about 12 requests and 72K input tokens. The Claude code is `_call_claude` in `takeoff/ai.py`.
   Gemini Flash-Lite (`gemini-3.5-flash-lite`) reads the sheet in overlapping zoomed-in sections, about 12 for a 36×42 sheet. For each room it reports the name and number, a point inside it, a rough box, whether it looks like a takeoff room (gym, fitness, training floor, studio, rubber or athletic flooring), and the finish tag. Then:
   - Exact outlines are traced from the drawing as usual, using the AI's rooms the same way as printed room labels. This is what names rooms on scanned plans, which have no text in the file.
   - Rooms the AI thinks belong in the takeoff are ticked.
   - If tracing can't isolate a room, its outline comes from the AI's box, snapped to the drawing, and is marked "⚠ check" in the list.
   - The message shows the estimated cost, usually 1–5¢ a sheet at $0.30 / $2.50 per million tokens. Only plan images are sent, not file or project names.
   Each section is one request, so a typical sheet is 12 requests. They're paced to 15 requests a minute, the free-tier limit; after a rate-limit pushback the app waits as long as Google asks (Claude's SDK backs off and retries by itself). On a paid tier with higher limits, start the app with `GEMINI_RPM=300` (or your limit) to run faster.
   **How a Claude trace works (three steps).** 1) Claude traces the room from a picture of the plan, told to follow wall faces round pilasters and piers, cross doorways in line with the wall, go behind equipment, ignore tags and pattern lines, and in an open plan follow the drawn edge line at any angle (`PROMPT` in `takeoff/ai_outline.py`). 2) The app fits that outline onto the drawing for precision (`_fit`: the floor between the drawn walls, square corners, finish lines on open edges, equipment notches checked). 3) Claude reviews: it sees the fitted outline drawn over the plan and either confirms it or gives a corrected outline (`review_outline`). A correction is fitted again; if that fit would move it noticeably (under 97% the same), Claude's correction is kept and only snapped onto the drawn lines beside its edges (`snap_to_lines`). Openings are found on the final outline. The review adds about 4–5¢ a room. In testing on 5 rooms, it fixed Fitness 422's top-right corner and confirmed the rest.
   **Claude traces the borders (Claude models only).** When the first pass runs on a Claude model, every room it ticks is then traced by Claude Opus 5.5 (`takeoff/ai_outline.py`, called from `_claude_borders` in `takeoff/server.py`). Claude sees a picture framed around the whole room, is told which room by its label and a dot inside it, and marks a rough outline. The exact border then comes from the drawing (`fill_outline`): the floor is filled out from Claude's outline across everything that isn't wall line work, so the border follows every wall face, including angled walls, piers, pilasters and window bays. Rules for the fill:
   - It may spread up to 6 ft past Claude's outline (bays or strips Claude cut across). Where it keeps going, it's running out through an opening, so that part is cut back to Claude's straight line.
   - Extra floor past Claude's outline is kept only if it opens wide onto the room: anything that joins through a gap under 8 ft (a doorway, a closet, a door swing) or is more than 5 ft deep stays out.
   - Corners stay square: the smoothing uses a square brush (a round one cut every corner round a pier off at 45°), and any bevel under 3" left by tracing the pixels is turned back into a sharp corner (`sharpen_corners`).
   - Notches narrower than about 12" close up, so the border runs straight past thin wall stubs and fins (ID111-L04's right wall). Pilasters and piers are wider, so it still steps round them.
   - Open-plan zones stop at their drawn edge: where nothing stopped the fill (an open edge, no wall), the app looks for a drawn line running alongside within 2.5 ft (a thin finish line, any colour) and fills again with it as the edge, so a zone in an open floor (Lifetime Sky's Free Weights 306 and 323) follows its own drawn boundary, diagonals included, instead of Claude's rough line. Those stretches also count as open edges for transitions (`_zone_lines`).
   - Equipment against a wall isn't wall: wherever the border cuts inside Claude's outline round something (over 0.75 SF), Claude Haiku looks at a close-up with it outlined and says whether it's structure (wall, pilaster, pier, column, shaft) or equipment standing on the floor (rack, shelf, bench, locker, machine). Equipment becomes floor and the border runs straight past it; structure stays (`check_notches`, about 0.03¢ each). Line weights can't tell them apart (BFT's treadmills are drawn with the walls' pen), but a picture can: in tests on BFT Training Floor 101 and Fitness 422, all 11 were right in both runs.
   - Where the two sides of a doorway don't line up, the border steps square instead of slanting across.
   - Tags aren't walls: window and door tags, finish tags and detail bubbles are often drawn right on a wall, and the border used to dip round them. Short thin lines that hug a word and surround it on all four sides (an oval, a hexagon, a bubble) are left out of the walls the fill sees (`tag_filter`); a line beside a label (a step edge, a finish line) still counts. Works on any plan with real text in the PDF (not scans).
   - A marked opening is always laid along one wall: if its two ends land on different edges and the line between them would run at an angle to the wall, both ends go on that wall's line (no slanted transitions).
   - A box with both diagonals drawn (an X: a shaft or opening in the floor) is treated as solid, so the fill can't leak into it through a gap in its wall.
   - Solid shapes thicker than 3 ft are ignored as walls (shaded zones and hidden fills, which Maroon Creek has), and furniture, tags and lines inside the room are filled over.
   - If the plan has no vector walls (a scan) or the fill comes out far from Claude's area, the old method is used: Claude's square edges snapped onto wall lines.
   **Doorways are squared to the wall.** Where the border crosses a doorway it first follows Claude's rough line, which can be a few degrees off and step round the jamb. The wall's own line is carried straight across instead: to where it meets the next wall, or on to the next stretch of the same wall (`square_openings`). And where the border bulges out past a confirmed opening (into a door swing), it's cut off along the jamb-to-jamb line (`cut_at_openings`).
   **Openings from the trace.** For each room Claude traces, the app gathers every stretch of the border that might be a way in: where the floor fill was cut off at a gap, what Claude marked as a way in while tracing (each end moved onto the border), and edges with no wall behind them (the rule-based search). Stretches along the same line that touch are joined into one. Each one is then shown to Claude Haiku 5.5 as a close-up with the stretch highlighted, and Haiku says what's there: door, open edge, window, wall or other (`check_openings` in `takeoff/ai_outline.py`). Only doors and open edges are kept, at their exact width from the drawing. Each check costs about 0.02¢. Find openings uses these for traced rooms and the rule-based search for the rest. An opening that's no longer on the border (after you join a storage room in) is dropped.
   On Maroon Creek (one test, 11¢ including tracing both gyms): all 4 of the drafter's gym openings were found at the right widths (the angled exterior door after a fix checked offline), the 2 storage room doors were kept (they drop out when the storage rooms are joined), and all 15 window and wall stretches were rejected. Check every suggestion anyway.
   Storage rooms stay as their own rooms: select the gym, then Shift-click the storage room to join them. About 4¢ and 15–20 seconds per ticked room, all at once. On Maroon Creek both gyms now follow the walls all the way round (Gym 1 96% overlap with the drafted outline, Gym 2 94%; the rest is the storage rooms the drafter joined in).
   Always review names, ticks and outlines before building: the AI can misread or miss rooms. The code is in `takeoff/ai.py`; change `MODEL` and the prices there to switch models.
2. **Choose rooms.** Click rooms on the plan or tick them in the list. Fix outlines with the tools on the left: drag corners, draw a room (D), click inside a room (A), or close a leaky doorway (B) and press **Detect rooms** again.
   **Rough outline for Claude (G).** Instead of a dot inside the room, click a few corners round it; double-click, Enter or click the first corner closes it, and you can outline several rooms. Enter with no outline open sends them. The outline is a guide, not the shape: it frames Claude's picture and is drawn on it in orange, and Claude is told it's only approximate and to trace the real wall faces (`HINT_TEXT` in `takeoff/ai_outline.py`). Then everything runs as for a dot: the fit onto the drawing, Claude's review, and the doorway checks. (Snapping the outline itself onto the walls was tried first and dropped: it kept every slanted or sloppy edge as drawn.) Tested on job 123's gym with an outline drawn 2 ft inside the walls and corners off by up to 1 ft (78% the same as the saved outline): Claude's trace 96%, after the review 97.1%, 1 doorway found, 74 s, about 16¢.
   **Symbols that aren't walls.** Two kinds of line work used to be read as wall, so the fitted border stepped round them (seen on IA310's fitness area): the "change in floor finish" X (drawn as two small filled V shapes: any small filled shape whose edges all run at about 45° is skipped, `_x_mark_piece` in `takeoff/detect.py`), and heavy dashed boundary lines drawn dash by dash (runs of 4+ equal short strokes in line, `_dash_strokes`; the floor leaked in through the gaps and left a sawtooth). An X under 2 ft across also no longer counts as a shaft. On the saved rooms (corners jittered 1 ft) the fill now matches 95.7% on average (93.9% before), worst room 92% (84%); Detect rooms is unchanged on them.
   **Walls in special print colours, walls beside hatched casework.** On IA310 the wall lines and the red PT01 finish are stored in a special colour that the PDF reader reports as white, so they were invisible to the fit; thin "white" shapes that print dark on the rendered sheet now count as wall (`_wall_mask_vector`). And a wall with hatched casework right behind it closes up into one block over 3 ft thick, which the fill clears as a shaded zone; it now keeps a thin rim at that block's edge, so the floor stops at the wall face instead of running through it. Room 19 now reaches the PT01 wall, stops at the face of the AF02 recess, and follows the curtain wall's inner line.
   **Lines, trim and pick, like AutoCAD.** **LINE (L):** click points one after another, or type a distance and press Enter to place the next point that far toward the cursor. Running object snaps with AutoCAD's markers and a label: endpoint (square), intersection (X), midpoint (triangle), perpendicular (from the last point), nearest (hourglass), on your lines, the drawing's horizontal/vertical lines and room outlines (Alt turns snapping off). F8 toggles ortho, Shift flips it while held. C closes back to the first point, U undoes the last segment, and Enter, Space, Esc or right-click ends the line. **TRIM (R),** AutoCAD's quick mode: every other line is a cutting edge (your lines, the drawing's, room outlines). Hovering shows the piece that would go (red dashes); click to trim it, or drag a fence across several pieces to trim them all. A line that crosses nothing is deleted. Shift-click extends a line to the next edge. Enter, Space, Esc or right-click ends. Then click inside with **A**: when your lines close off a shape around the click (gaps of a few screen pixels count as closed), the room is exactly that shape. Where they don't, A finds the room from the walls as before, with your lines acting like closed openings.

   **Drawing a room (D).** Click the room's corners: each edge is a straight line between your clicks, and corners snap to the drawing's lines. Shift = square edge, Alt = free point with no snapping, Cmd/Ctrl-Z undoes the last corner. (Edges used to follow the wall face between clicks; that was removed because it wrapped round symbols and tags drawn on walls.)
   **Fix outlines precisely.** Zoom in (scroll, or the + button): a sharp, full-resolution view of the plan loads so you can see exact wall lines.
   - Drag a corner of the selected room to move it. Corners **snap to the drawing's own wall lines** (a green square shows the snap). Hold **Alt** to place freely, or **Shift** to keep the edge square.
   - **Double-click an edge** to add a corner. **Alt-click a corner** to delete it.
   - Room outlines are snapped onto the PDF's wall lines automatically when rooms are detected, so corners are exact rather than traced.

   **Columns.** Free-standing columns (solid, roughly square, up to 4 ft) inside a room are found automatically and cut out of its area. They show as holes in the room. Press **K** and click two opposite corners to add a column (corners snap to the drawn column). For a **round column**, press **O** and click inside it: a circle drawn on the plan is found and cut out exactly (shown highlighted as you hover). If it isn't drawn as a circle (a scan, for example), click its centre and then its edge. For anything irregular (an odd-shaped column, a planter, casework), press **E** (Exclusion) and click its corners like drawing a room: double-click, Enter or click the first corner to finish. Points snap to the drawing, Shift keeps edges square, Cmd/Ctrl-Z undoes a point. A convex exclusion that runs past the room's edge is trimmed to the room, so only the part inside is taken out. **Alt-click inside a column** to remove it. The sheet PDF leaves columns out of the fill and tile grid, and the BOM CSV lists the columns and their SF per room.

   **Different buildups in one room.** Press **X** (Split) and click across the area: start at one wall, click at the opposite wall (add clicks in between for a bend), then double-click or press Enter. Points snap to the drawing; Shift keeps the cut square. The area becomes two, for example "FITNESS 422 · A" and "· B". Their areas add up to the original, and each column stays with the piece it's in. Each piece gets its own buildup in step 3.

   **Check measurements.** Select a room to see the length of each side, and compare them with the plan's dimension strings. Use the **measure tool (M)** to click along anything on the plan: each segment shows its length in feet and inches, with a running total. Double-click or press Enter to finish. Click back on the first point to close the shape and get its area and perimeter. After any click you can **type a length** (`12'6`, `12-6`, `42"` or `12.5`; plain numbers are feet) and press Enter: the next point goes that far toward the cursor (snapped square when you're within 8°). Hold Shift for straight lines, press Esc to cancel, and press Esc again to clear all measurements. If a known dimension measures wrong, the scale is off: fix it at the top, or calibrate (C) on that dimension.
   When the right rooms are ticked, click **Finalize Room Selections**. The Rooms step folds into a bar (click it to go back), and only the selected rooms stay on the plan.
3. **Buildups, one room at a time.** The first selected room opens with a **Buildup** stack. The top dropdown is the top layer (the walking surface) and the last is the layer on the slab. Pick a product for each layer from the full catalog in `config/products.yaml`; products are never grouped into systems. **+ Add layer below** adds a layer, ↑ ↓ reorder, ✕ removes. Each layer shows whether it's glued or loose to the layer below (the product's normal bond; change it for this room if needed, for example to glue plywood) and the total buildup height. **Copy** reuses another room's buildup, and a checkbox appears when there's a finished floor whose adhesive Pliteq supplies. **Next Room** moves on; finished rooms fold into bars you can click to edit. Each layer can also be given a **colour** (A–E): each colour is ordered as its own BOM line with its own overage, for example "1400 SF GenieMat FIT70 (Colour A) (350 tiles) (9% overage)", and gets its own callout. **Concrete Topping** gets a **Thickness** box (mm) for that room, since it varies by job; it counts toward the buildup height (blank = unknown, as before) and is written in the callout, e.g. "2" Concrete Topping (supplied by others)" (nearest ⅛"). Concrete topping over an FF product is taken as flush with the floor around it, so the transitions only make up the layers on top of the concrete (none on top = no ramp). Any product marked `thickness_input: true` in `config/products.yaml` gets the same box. **Custom layer** (bottom of the product list) is for anything not in the catalog: type its label, exactly as it should print in the callout, and its thickness in mm. It counts toward the buildup height (and so the transitions), is never in the BOM, and is never hatched: the sheet hatches the top-most Pliteq layer instead. Overage isn't entered: it comes from each product's total across the takeoff (`config/rules.yaml`: 15% up to 250, 13% up to 500, 11% up to 750, then 9%; exactly on a boundary gets the higher one).
4. **Sheet.** Shows the BOM, the notes that will print (chosen automatically from `config/notes.yaml`), and **Add notes** for the ones the drafter decides. Enter the project, drawing number and drawn-by, then **Build takeoff sheet**. The PDF opens in a new tab, and `jobs/<drawing no>/` holds the PDF, the BOM CSV and `rooms.json`. Load that `rooms.json` later to pick up where you left off. On the sheet, the selected rooms are drawn over the plan (faded 70%, without markups), and the plan is turned a quarter turn when that draws it more than 8% larger (Maroon Creek: 13%). The plan may reach up beside the BOM, as on the drafted sheets, wherever no room or ramp would run into it.

Everything runs on your computer. Nothing is uploaded anywhere.

The command-line steps below do the same thing and are still useful for batch runs.

## 1. Set up (once)

You need Python 3.10 or newer.

```bash
cd takeoff-pipeline
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_engine.py        # should print "All sample takeoffs match."
```

## 2. Detect rooms

```bash
python -m takeoff detect "ID111 Finish Plan.pdf" -o work/
```

This reads the plan scale from the sheet (for example `1/8" = 1'-0"` next to the plan title), traces every enclosed room, and names rooms from the text in the PDF (`FITNESS 422`, `YOGA 420`).

It writes two files:

- `work/rooms.json`: every room's outline (in PDF points), name and area.
- `work/rooms_preview.pdf`: the plan with every room outlined and numbered. Open it to see which number is which room.

Useful options:

| Option | What it does |
|---|---|
| `--page 3` | Use another page of a multi-page set |
| `--scale 1/4` | Set the scale if it can’t be read or is wrong |
| `--gap-ft 5` | Bridge wider openings (default 3.75 ft) |
| `--threshold 180` | Wall line darkness, 0–255 (lower keeps only darker lines) |
| `--min-sf 50` | Drop small spaces such as closets |
| `--barriers rooms.json` | Close doorways you drew in the review page, then detect again |

## 3. Choose the rooms

Pick one of three ways.

**By name:**
```bash
python -m takeoff select work/rooms.json --names FITNESS,YOGA
```

**By number from the preview:**
```bash
python -m takeoff select work/rooms.json --ids 10,11,12
```

**At a prompt:**
```bash
python -m takeoff select work/rooms.json -i
```

**In the browser, to fix outlines:** open `review/review.html` in Chrome, Edge or Firefox (double-click it). Then:

1. Open the plan PDF.
2. Click **Load rooms.json**.
3. Click rooms to include them.
4. Drag corners to fix outlines, draw any missing rooms (D), or close leaky doorways (B).
5. Click **Save rooms.json** and use the saved file in the next step.

The page needs an internet connection only to load its PDF reader and font.

## 4. Build the takeoff

```bash
python -m takeoff build work/rooms.json \
  --system FIT70 \
  --project "Alyiah Sloans Lake" --dwg ASH-1 --drawn-by SF \
  --standard-lf 0 --ada-lf 0 \
  -o out/ASH-1.pdf
```

This writes `out/ASH-1.pdf` (the takeoff sheet) and `out/ASH-1.csv` (the BOM plus each room’s area). Areas are recomputed from the outlines, so edits made in the review page are reflected.

- **Buildups:** `--buildup FIT70/FFNC50NBN` (top first) for rooms without one in rooms.json. List product codes with `python -m takeoff products`.
- **Overage:** `--overage 13` overrides the overage for every layer.
- **Transitions:** in the Sheet step, set the slab to **Not recessed** (or **Partly recessed** with the recess depth). **Recessed** means fully recessed: no transitions, and no depth to enter. **Find openings** suggests the openings with no wall; **Mark one (T)** adds a doorway or an edge where two buildups meet: click both ends for a custom width, or **double-click on a room's edge** for a standard 42" ADA opening centred there (slid along to stay on the edge, so near a corner it sits flush to it). After the first click you can also type the width (`42"`, `3'6`) and press Enter. For each opening, choose Standard or ADA, where the ADA ramp sits (centred, flush to one wall, or between walls), and whether the door swings in (FIT08 landing). The rules are in `takeoff/ramps.py`:

  **Editing on the plan.** Click a transition, a measurement or a closed doorway (barrier) to select it. This works with the select tool, and with the transition, measure and barrier tools before you start a new one. A small box appears beside it: for a transition, type a new length and press Enter, switch Standard/ADA, or press **42"** for a standard ADA width. Drag an end to resize it along its wall, or drag the middle to slide it. **Delete** (or Backspace) removes the selected item, Esc deselects, and **Cmd/Ctrl-Z** undoes adds, moves, resizes and deletes. The Transitions list in the panel also has a length box on each row.

  - **Height** = the room's buildup thickness, less the buildup on the other side or the recess.
  - **Side:** between two buildups a transition sits on the lower room's side; onto bare slab it sits inside the raised room. The editor shades where each transition will sit (teal standard, red ADA, green landing), from the latest takeoff. To put one on the other side (e.g. a closet door would swing into it), select it and click **⇅ Side**: between buildups it moves onto the higher room's floor (its SF comes off that room); onto slab it moves out onto the slab (no SF comes off the room). Saved with the job (`flip` on the opening).
  - **Ends:** an end up against a wall is cut square with no wing. That's judged from the plan itself (dark wall drawn just past the end, 3" and 8" out, across the ramp's depth; a door leaf standing open is too thin to count; hidden fills don't count because it uses the rendered page), so it works even where a room is drawn over the walls; or from the room's outline turning into the room within 3" (an inside corner, or the side of a door recess). The same check takes the ADA wing off a side that's against a wall, but only where the ramp is between the walls: the app measures how far the wall (or the room's edge) runs down each side from the doorway, leaves the wing out there and starts it where the wall ends (ordered by the length past the wall). A side walled the whole way has no wing; a ramp that only just nestles into a doorway keeps nearly all of its wings. Each depth is checked every inch out to 4" from the ramp's side, and a black wall line crossed in that 4" counts as wall too (walls drawn as two thin lines have a white gap a single point can miss). A flipped ramp, out of the room onto the slab, is measured the same way, with the open floor outside the room. Any other end gets the 45° cut with its triangular wing. A 1" jog in a traced outline doesn't count as a wall.
  - **Between two buildups** the ramp runs off the higher one onto the lower room's floor: it's drawn in the lower room (in the doorway) and its SF comes off that room. If the higher room has an underlayment the lower one doesn't, the callout says the ramp sits over it, e.g. "GenieMat FIT30 Standard Transition over GenieMat FFNC25NBN" (Cottages Cardio). That underlayment carries on under the ramp, so the ramp's footprint is added to its SF and adhesive, and when it's an FFNC NBN the "FFNC25NBN to be used for the tiered ramping and support for the tile edge" note replaces the wood-support note.
  - **Ramp type:** FIT30 ramps up to 30 mm. FIT70 ramps up to 70 mm, trimmed to the buildup height (a trimmed ramp is cut down, so it's drawn shorter in proportion: 45 mm = about 21" instead of 32½"), with the shimmed/trimmed note. Above that, height ÷ 70 rounded up FIT70 tiers in a row: the first, at the door, sits on the floor and is trimmed to what's left over; the rest are full tiers over wood blocking (Visa 178 mm = 1 + 2x over blocking).
  - **ADA openings:** one 42" ADA ramp, with standard wings wall to wall and on its exposed sides.
  - **Standard pieces:** 48" each. Every standard length, straight edges and the triangular wings beside an ADA ramp alike, rounds up to the next 6" before the sharing rule (FIT30 wings: 13" -> 18", so two to a piece; FIT70 wings: 32.5" -> 36", one each). Edges of 42" or less share a piece, up to 3 per piece and 42" in total. A run with a 45° cut-back at one or both ends (ending at a door jamb) takes one extra piece per tier for its triangles (Cottages Cardio: 6 + 2 pieces of wings + 2 runs with triangles + 1 spare = 11; Modera Mystic: 6 + 2 door ramps + 1 = 9). A run end only counts as a door jamb when the room's edge carries straight on past it for at least a foot; if the outline turns within that foot (a corner, or round the end of a wall pier), the triangle is drawn but not ordered (Maroon Creek: 44 LF however the outline is traced). `tests/fixtures/` holds the Cottages, Modera and Maroon Creek jobs; the tests check them against the drafters' sheets. `transition_overage` spares are added; ADA ramps get none.
  - **Joins and corners** (drawn as on the Lifetime Sky LFS-3 DWG): runs of standard ramp along a room's edge are drawn as one strip, mitred at a square corner; at a slanted wall the straighter run ends square and the slanted one meets it. Beside an ADA ramp the strip is cut at 45° from the ADA ramp's wall corner, and the ADA ramp's wing on that side runs from its far corner to the strip's inner edge. On a side with no standard run, the wing is a right triangle lying along the wall, one standard depth long (Cottages Cardio). A run that ends right in an inside corner (within 1½") butts square into the wall; a run ending at a door jamb, even a few inches short of a corner, gets the 45° cut-back with its triangle piece. An ADA ramp that ends at a corner of the room gets no wing on that side (it would hang outside the room), and isn't ordered one. Where a wing runs past the corner of a wall that turns away right beside the jamb (a slanted wall), the wing keeps its standard shape and another standard piece joins the part past the corner back to the wall (square to the doorway, out to the wall); it's ordered as a short standard edge, its length along the doorway. An ADA doorway up to 45" wide is still one 42" ramp, trimmed to fit (Cottages Cardio: 44.4" = 1 ADA).
  - **Room SF:** the ramps (and any FIT08 landing) sit inside the room you drew to the walls, so their footprint comes off that room's SF, as on the drafted sheets (Lifetime Sky LFS-3: 227 SF of tiered ramps). The BOM CSV lists it per room as `transitions_sf`. On the sheet, a takeoff with one buildup is left unfilled; colour fills only appear when there are several buildups to tell apart.
  - **Tiers:** every tier is counted in the LF. Each tier's ADA ramp gets standard wings down both sides, as long as the ADA ramps up to it, and each tier gets its own strip down the sides of a FIT08 landing.
  - **Adhesive (FAS1):** LF × depth/12 ÷ 500 (32.5" FIT70 ADA, 18" FIT70 std, 13" FIT30 ADA, 12" FIT30 std).
  - **FIT08 pad adhesive:** a FIT08 area that takes a single roll is glued on the whole 100 SF roll (0.2 pails), like a landing (Parkview, Modera Mystic); bigger FIT08 areas use net SF + overage.
  - **Adhesive total:** amounts quoted in the notes (e.g. "1.7 pails … for GenieMat FFNC25NBN to slab") are added to the BOM total as quoted, to the tenth, then rounded up to whole pails, as the drafters do (Cottages Cardio: 1.7 + 1.19 + 0.12 = 3.01 → 4).
  - **Client requests:** the dates go into the client-request notes.

### All in one

```bash
python -m takeoff run plan.pdf --names FITNESS --system FIT30 --project "OPP Red Lake" --dwg RLM-1
```

If you don't pass `--names` or `--ids`, it asks which rooms to include.

## Configuring products and the title block

- `config/products.yaml`: every product a buildup can use: code, display name, family, unit and unit size, thickness, adhesive class and normal bond. Products without a unit size are left out of the dropdowns until a size is added (plywood and finished floor are always offered; they're never in the BOM).
- `config/rules.yaml`: overage tiers, adhesive coverage and factors, ramp sizes, sharing, overage and adhesive depths (transitions).
- `config/notes.yaml`: the notes library. `auto: true` notes are added by the engine (`takeoff/engine.py`); `auto: false` notes are ticked by the drafter under **Add notes**.
- `config/sheet_template.json`: the sheet's border, title block, Pliteq logo, fixed text and field positions, made from Pliteq's AutoCAD template. To use a new template, save it as DXF and run `python -m takeoff.template "path/to/template.dxf"` (needs `pip install ezdxf`). Text is set in Georgia, as in the template.
- `config/company.yaml`: the reviewer's initials for REVIEWED BY (the rest of the title block now comes from the template).
- `config/legacy/systems_products.yaml`: the earlier system-based catalog, kept for reference only.

### How quantities and adhesive are worked out

- Each product's quantity is its total across all selected rooms (SF, or room perimeter in LF for PMI12PF). Overage comes from that total; units are rounded up.
- Adhesive is worked out room by room, for each glued layer onto the layer below (or the slab), on the ordered area (net + overage). FAS3 for plank-class products; FASHM for a rolled layer on FIT tiles; FAS1 for everything else (FASHM-equivalent pails × 0.3). Gluing onto FF/FFNC uses 1.515× the area, gluing onto plywood adds area/1500, FIT on FFxxNBN is glueless (on FFNCxxNBN it is still glued). Pails are summed per adhesive and rounded up to whole pails (at least 1); the pails quoted in notes are rounded to the nearest tenth. FFNC50NBN is two plies of FFNC25NBN, so it is glued to the slab and ply to ply (the notes quote only the slab layer).
- Multi-ply FF (products.yaml `plies`): FFNC50 = 2 x FFNC25, FFNC50NBN = 2 x FFNC25NBN, FFNC75NBN = 3 x FFNC25NBN. Every ply is glued (to the slab and ply to ply).
- Systems (products.yaml `components`, top first): FIT62 = FIT30 on FFNC32, FIT55 = FIT30 on FFNC25, PeakForce = Envy (FAS3) on RST10 (FASHM) on FIT30 (FAS1), PeakForce ELITE = the same on FIT70. The BOM lists the components; the callout shows the system name.
- Perimeter isolation (PMI12PF) is added automatically, on the room perimeter in LF with the standard overage tiers on the total, for rooms whose buildup has plywood, FF70 (FF70, FF70LDM, FF70-LF) or FF under a concrete topping. It is not picked as a layer.
- A FIT tile with any layer on top of it is listed as the -C1 tile (e.g. GenieMat FIT70-C1), as its own BOM line.
- `tests/test_engine.py` checks the engine against past sheets (Alyiah, 175 3rd St, Westerly, RST10 on FIT70). Add a case when you finish a real job.

## How detection works, and when it struggles

**Shaded floors.** Some plans tint the scope area grey (I1-01 Level 2's fitness centre). Any solid area thicker than 3.5 ft can't be a wall, so Detect rooms clears it and keeps its edge as the room's boundary (`drop_thick` in `takeoff/detect.py`). A narrow neck in a shaded room can still split it in two; Shift-click to join.

**PDFs exported from CAD/Revit (most plans):** walls are taken from the PDF's own line work, not from a picture of the page, and checked against the room labels printed on the plan (`takeoff/rooms.py`).

1. **Text is never drawn**, so room names, tags, dimensions and notes can't be mistaken for walls. Coloured markups (red clouds) are skipped too.
2. **Three wall rules are tried, and combined.** Some architects draw walls with the heaviest lines (ID111). Others draw them in a medium weight (glass office fronts on Topridge), or draw equipment heavier than walls (the BFT sheet), where only long straight lines count. The rule that separates the most numbered rooms is the base, and any room another rule separates on its own is taken from that rule.
   - **Background tones.** A big plain grey shape (a background sheet or shaded zone) counts only by its edge, and a big white fill laid over earlier line work paints it out, the way it looks on paper. Before this, plans exported with a grey background (Topridge) traced as one solid block.
   - **Notes, legends and title-block boxes** (outlines holding 40+ words that cover a fifth of the box) are dropped.
3. **Room labels are read in any direction**, including sideways and multi-line names (`TRAINING FLOOR 101`, `JAN. CLOSET 104`), and numbers such as `918`, `900N`, `B102` and `10-005`. When a sheet shows the same room twice (a power plan above a finish plan), each copy is traced on its own.
4. **Openings are closed step by step**, from about 2.8 ft up to about 11 ft. Each numbered room takes the narrowest setting at which it holds only its own number. That separates rooms joined by wide openings, such as a lobby storefront or Yoga opening into Fitness, then gives each room back its full shape.
5. **Rooms that open into each other with no door** (a lounge, corridor and copy area) come out as one area named after all of them, marked to check: split it (X) or close the opening (B) if only part is in the takeoff.
6. **Spaces without a number**, such as corridors, are kept and named from their finish tag if there is one. Areas far from the numbered rooms (title block, legend, details) are dropped.
7. **Exact corners and columns.** Each traced edge that runs square to the sheet is moved onto the nearest wall line in the PDF, and corners are rebuilt as exact intersections. Free-standing solid columns are cut out of the room's area.
8. **Colour-fill plans.** Where the architect drew a room as a filled shape (Revit colour fills), that shape is used for any numbered room the tracing left merged or missed (Westchester's FITNESS 136).
9. **Scrambled fonts.** Some CAD exports store text with every character code shifted (the sheet shows EXIT, the text layer holds `(;67`). Each font's shift is found from common plan words and undone, so room names and the scale read normally (Lifetime Sky).
10. **Scale.** Metric ratios (1:50, 1:100) are read too; 1:250 and up are taken as key or site plans and ignored. The scale is read from the sheet's scale note. If that isn't stored as text, it is estimated from dimension strings such as `7'-0"` and the lines they sit on, and the app says "estimated from dimensions, check it".

Results:

- **ID111:** FITNESS 422 2,561 SF and YOGA 420 301 SF. The hand-drawn ASH-1 takeoff has 2,812 SF for the two together.
- **BFT** (¼″ scale, read from dimensions): TRAINING FLOOR 101 at 1,280 SF and LOBBY 100 at 140 SF. Its restrooms, stairs and closet aren't found, because their walls aren't drawn as long lines. Add them with A or D if you need them.

**Scanned plans** have no line work or text layer. They fall back to tracing the image, where the "wall line darkness" and "thin lines" settings apply, rooms come out unnamed, and the scale must be set by hand.

It still struggles with:

- **Wide openings** (over about 3.75 ft) between a room and a corridor. Close them with a barrier line (B) and detect again.
- **Plans drawn with unusual line weights.** The wall cut-off is 0.5 pt for the first pass and 0.3 pt for the second (`min_stroke_pt`, `fine_stroke_pt` in `detect.py`).

**Click inside a room (A)** uses the same tracing on the server: the first click on a page takes a few seconds while the page is traced, later clicks are instant. If nothing detected holds the click, it traces outward from that spot, narrowest openings first, and keeps the first enclosed space with at most one room number. This is the quickest way to take off one or two rooms: you don't need Detect rooms first.

Always check the outlines before ordering. The sheet's general notes say the same.

## Project layout

```
takeoff/
  scale.py    find and parse the drawing scale
  detect.py   room detection (OpenCV), geometry helpers
  labels.py   room names from the PDF text layer
  engine.py   buildups -> BOM lines, overage, adhesive, notes (config/*.yaml)
  rules.py    BOM line formatting
  sheet.py    takeoff sheet + numbered preview PDFs (PyMuPDF)
  cli.py      detect / select / build / run / products / serve
  server.py   local web app (python3 -m takeoff serve)
  ai.py       optional AI first pass with Gemini or Claude
config/       products.yaml, rules.yaml, notes.yaml, company.yaml
app/          index.html: the web app's page
review/       review.html: older standalone review page
jobs/         takeoffs built from the app
tests/        regression cases from real takeoff sheets
examples/     output from the Aliyah Sloans Lake level 4 plan
```

## Ideas for next steps

- **Batch runs:** add a `jobs.yaml` listing many plans and their systems.
- **AI review:** a step that reads the finish legend and spec notes and suggests the system per room.

/* ═══════════════════════════════════════════
   NextPlay — Application Logic (Redesign)
   ═══════════════════════════════════════════ */

const state = { page:'home', stats:null, selectedAlgo:'Hybrid', selectedMood:null, selectedBracket:null, tasteGames:[], homePage:1 };
const $=(s,c=document)=>c.querySelector(s);
const $$=(s,c=document)=>[...c.querySelectorAll(s)];
const content=()=>$('#content');

// ── Helpers ──
async function api(path,opts={}){ const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts}); if(!r.ok) throw new Error(`API ${r.status}`); return r.json(); }
function debounce(fn,ms){ let t; return(...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms);}; }
function escHtml(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function formatNum(n){ if(n>=1e6) return (n/1e6).toFixed(1)+'M'; if(n>=1e3) return (n/1e3).toFixed(1)+'K'; return n.toLocaleString(); }

const PLACEHOLDER='data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="460" height="215" fill="%23f5f5f5"><rect width="460" height="215"/></svg>');
function imgSrc(url){ return url && url.startsWith('http') ? url : PLACEHOLDER; }
function httpsUrl(url) { return url ? url.replace(/^http:\/\//i, 'https://') : ''; }

// ── Theme ──
function initTheme(){
  const saved = localStorage.getItem('nextplay-theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
}
function toggleTheme(){
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('nextplay-theme', next);
  if(state.page==='trending') renderTrending(); // re-render chart for colors
}

// ═══════════════════════════════════════════
// REUSABLE COMPONENTS
// ═══════════════════════════════════════════

function renderGameCard(g, onclick){
  const genres = (g.genres||'').split(',').slice(0,2).join(', ');
  const year = (g.release_date||'').match(/\d{4}/)?.[0]||'';
  const meta = [g.developer, year].filter(Boolean).join(' • ');
  const card = document.createElement('div');
  card.className = 'game-card';
  card.innerHTML = `
    <div class="game-card-img-wrapper">
      <img class="game-card-img" src="${imgSrc(g.header_image)}" alt="${escHtml(g.name)}" loading="lazy">
    </div>
    <div class="game-card-body">
      <div class="game-card-title">${escHtml(g.name)}</div>
      <div class="game-card-meta">${escHtml(meta)}</div>
      <div class="game-card-genres">${escHtml(genres)}</div>
    </div>`;
  if(onclick) card.addEventListener('click', ()=>onclick(g));
  return card;
}

function renderRecCard(g, rank){
  const genres = (g.genres||'').split(',').slice(0,3).join(', ');
  const year = (g.release_date||'').match(/\d{4}/)?.[0]||'';
  const meta = [g.developer, year, genres].filter(Boolean).join(' • ');
  const score = g.match_score != null ? g.match_score+'%' : '';
  const steamLink = g.steam_link || `https://store.steampowered.com/app/${g.appid}`;
  const div = document.createElement('div');
  div.className = 'rec-card';
  div.innerHTML = `
    <img class="rec-card-img" src="${imgSrc(g.header_image)}" alt="${escHtml(g.name)}" loading="lazy">
    <div class="rec-card-body">
      <div class="rec-card-top">
        <div class="rec-card-title">${rank ? rank+'. ':''}${escHtml(g.name)}</div>
        ${score ? `<span class="rec-card-score">${score} Match</span>` : ''}
      </div>
      <div class="rec-card-meta">${escHtml(meta)}</div>
      ${g.explanation ? `<div class="rec-card-reason">${escHtml(g.explanation)}</div>` : ''}
      <div class="rec-card-actions">
        <button class="btn btn-sm btn-secondary" onclick="navigateToExplore('${escHtml(g.name).replace(/'/g,"\\'")}')">Deep Dive</button>
        <a class="btn btn-sm btn-ghost" href="${steamLink}" target="_blank" rel="noopener">Steam ↗</a>
      </div>
    </div>`;
  return div;
}

function renderSearchBox(id, placeholder, onSelect){
  const wrapper = document.createElement('div');
  wrapper.className = 'search-wrapper';
  wrapper.innerHTML = `
    <svg class="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
    <input type="text" class="search-input" id="${id}" placeholder="${placeholder}" autocomplete="off">
    <div class="autocomplete-list" id="${id}-ac"></div>`;
  
  const input = wrapper.querySelector('.search-input');
  const acList = wrapper.querySelector('.autocomplete-list');

  const doSearch = debounce(async(q)=>{
    if(q.length<2){ acList.classList.remove('open'); return; }
    try {
      const results = await api(`/api/games?q=${encodeURIComponent(q)}&limit=10`);
      if(!results.length){ acList.classList.remove('open'); return; }
      acList.innerHTML = results.map(g=>`
        <div class="autocomplete-item" data-name="${escHtml(g.name)}" data-img="${escHtml(imgSrc(g.header_image))}">
          <img class="autocomplete-thumb" src="${imgSrc(g.header_image)}" alt="" loading="lazy">
          <div class="autocomplete-text">
            <div class="autocomplete-name">${escHtml(g.name)}</div>
            <div class="autocomplete-meta">${escHtml((g.genres||'').split(',')[0])}</div>
          </div>
        </div>`).join('');
      acList.classList.add('open');
    } catch(e){ console.error(e); }
  }, 250);

  input.addEventListener('input', ()=>doSearch(input.value.trim()));
  input.addEventListener('focus', ()=>{ if(acList.children.length) acList.classList.add('open'); });
  
  // Handle click on autocomplete item
  acList.addEventListener('mousedown', e=>{
    // mousedown fires before blur
    const item = e.target.closest('.autocomplete-item');
    if(!item) return;
    const obj = { name: item.dataset.name, img: item.dataset.img };
    input.value = obj.name;
    acList.classList.remove('open');
    if(onSelect) onSelect(obj);
  });
  
  input.addEventListener('blur', ()=>setTimeout(()=>acList.classList.remove('open'), 150));
  
  return wrapper;
}

function skeletonCards(n=6){ let h='<div class="game-grid stagger">'; for(let i=0;i<n;i++) h+='<div class="skeleton skeleton-card"></div>'; return h+'</div>'; }
function skeletonList(n=5){ let h='<div class="rec-list">'; for(let i=0;i<n;i++) h+=`<div style="display:flex;gap:20px;padding:20px;border-bottom:1px solid var(--border)"><div class="skeleton" style="width:160px;height:80px;border-radius:4px"></div><div style="flex:1"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line" style="width:40%"></div></div></div>`; return h+'</div>'; }

// ═══════════════════════════════════════════
// PAGES
// ═══════════════════════════════════════════

// ── HOME ──
async function renderHome(){
  state.homePage = 1;
  const el = content();
  el.innerHTML = `
    <div class="page-header fade-in">
      <h1 class="page-title">Explore Games</h1>
      <p class="page-subtitle">Your next favorite game is waiting.</p>
    </div>
    <div id="home-stats" class="stats-row fade-in"></div>
    <div class="section-title fade-in">Popular Right Now</div>
    ${skeletonCards(8)}`;

  try {
    const [stats, popular] = await Promise.all([api('/api/stats'), api('/api/popular?limit=12')]);
    state.stats = stats;
    $('#home-stats').innerHTML = `
      <div class="stat"><div class="stat-value">${formatNum(stats.n_games)}</div><div class="stat-label">Games</div></div>
      <div class="stat"><div class="stat-value">${formatNum(stats.n_users)}</div><div class="stat-label">Users</div></div>
      <div class="stat"><div class="stat-value">${formatNum(stats.n_interactions)}</div><div class="stat-label">Interactions</div></div>
      <div class="stat"><div class="stat-value">${stats.avg_playtime_hrs}h</div><div class="stat-label">Avg Playtime</div></div>`;

    const grid = document.createElement('div'); grid.className='game-grid stagger'; grid.id='popular-grid';
    popular.results.forEach(g=>grid.appendChild(renderGameCard(g, game=>navigateToExplore(game.name))));
    const existing = el.querySelector('.game-grid');
    if(existing) existing.replaceWith(grid);

    if(popular.has_more){
      const lm = document.createElement('div'); lm.className = 'load-more-container';
      lm.innerHTML = `<button class="btn btn-secondary" id="load-more-btn">Load More</button>`;
      el.appendChild(lm);
      $('#load-more-btn').addEventListener('click', loadMore);
    }
  } catch(e){ console.error(e); }
}

async function loadMore(){
  const btn=$('#load-more-btn'); if(btn){btn.disabled=true;btn.textContent='Loading...';}
  state.homePage++;
  try {
    const pop = await api(`/api/popular?limit=12&page=${state.homePage}`);
    const grid = $('#popular-grid');
    if(grid) pop.results.forEach(g=>{ const c=renderGameCard(g,game=>navigateToExplore(game.name)); c.classList.add('fade-in'); grid.appendChild(c); });
    if(pop.has_more && btn){btn.disabled=false;btn.textContent='Load More';} else if(btn) btn.parentElement.remove();
  } catch(e){ console.error(e); if(btn) btn.textContent='Error'; }
}

// ── DISCOVER ──
async function renderDiscover(){
  const el = content();
  el.innerHTML = `
    <div class="page-header fade-in">
      <h1 class="page-title">Discover</h1>
      <p class="page-subtitle">Find games similar to one you already love.</p>
    </div>
    <div class="search-section fade-in">
      <div id="discover-search"></div>
    </div>
    <div class="tab-row fade-in" id="algo-tabs"></div>
    <div id="discover-results"></div>`;

  const sb = renderSearchBox('discover-input','Search for a game...', obj=>doRecommend(obj.name));
  $('#discover-search').appendChild(sb);

  const tabs = $('#algo-tabs');
  ['Hybrid','Content-Based','Collaborative'].forEach(algo=>{
    const btn = document.createElement('button');
    btn.className = 'tab-btn'+(algo===state.selectedAlgo?' active':'');
    btn.textContent = algo;
    btn.addEventListener('click', ()=>{
      state.selectedAlgo = algo;
      $$('.tab-btn',tabs).forEach(b=>b.classList.toggle('active',b.textContent===algo));
      const inp=$('#discover-input'); if(inp&&inp.value.trim()) doRecommend(inp.value.trim());
    });
    tabs.appendChild(btn);
  });
}
async function doRecommend(name){
  const r=$('#discover-results'); r.innerHTML=skeletonList(6);
  try {
    const d = await api('/api/recommend',{method:'POST',body:JSON.stringify({game:name,algorithm:state.selectedAlgo,count:12})});
    if(!d.results||!d.results.length){ r.innerHTML=`<div class="empty-state">No matches found for "${escHtml(name)}".</div>`; return; }
    r.innerHTML=`<div class="section-title" style="margin-top:16px;">${d.results.length} recommendations for <strong>${escHtml(name)}</strong></div>`;
    const list=document.createElement('div'); list.className='rec-list stagger';
    d.results.forEach((g,i)=>list.appendChild(renderRecCard(g,i+1))); r.appendChild(list);
  } catch(e){ r.innerHTML=`<div class="status-msg error">Something went wrong.</div>`; }
}

// ── MOOD ──
async function renderMood(){
  const moods = [
    {key:'action',title:'Action & Intensity',desc:'Shooters, fighters, and high-octane games.'},
    {key:'relaxing',title:'Calm & Contemplative',desc:'Puzzles, simulations, and cozy experiences.'},
    {key:'competitive',title:'Competitive & Esports',desc:'Multiplayer, MOBA, and ranked play.'},
    {key:'story',title:'Story-Driven',desc:'Deep narratives, RPGs, and adventures.'},
    {key:'horror',title:'Horror & Suspense',desc:'Survival horror and psychological thrillers.'},
    {key:'strategy',title:'Strategy & Brain',desc:'Grand strategy, city builders, and tactical.'},
    {key:'retro',title:'Retro & Arcade',desc:'Pixel graphics, classics, and arcade.'},
  ];
  const el = content();
  el.innerHTML = `
    <div class="page-header fade-in">
      <h1 class="page-title">Mood</h1>
      <p class="page-subtitle">What kind of experience are you in the mood for?</p>
    </div>
    <div class="mood-grid fade-in" id="mood-grid"></div>
    <div id="mood-results"></div>`;
  const grid = $('#mood-grid');
  moods.forEach(m=>{
    const card = document.createElement('div');
    card.className = 'mood-card'+(state.selectedMood===m.key?' selected':'');
    card.innerHTML = `<div class="mood-card-title">${m.title}</div><div class="mood-card-desc">${m.desc}</div>`;
    card.addEventListener('click', ()=>{ 
      state.selectedMood=m.key; 
      $$('.mood-card',grid).forEach(c=>c.classList.remove('selected')); 
      card.classList.add('selected'); 
      loadMood(m.key); 
    });
    grid.appendChild(card);
  });
  if(state.selectedMood) loadMood(state.selectedMood);
}
async function loadMood(key){
  const r=$('#mood-results'); r.innerHTML=skeletonList(6);
  try {
    const d = await api(`/api/mood/${key}`);
    if(!d.results||!d.results.length){ r.innerHTML=`<div class="empty-state">No games found.</div>`; return; }
    r.innerHTML = '';
    const list=document.createElement('div'); list.className='rec-list stagger';
    d.results.forEach((g,i)=>list.appendChild(renderRecCard(g,i+1))); r.appendChild(list);
  } catch(e){ r.innerHTML=`<div class="status-msg error">Failed to load.</div>`; }
}

// ── MY TASTE ──
async function renderTaste(){
  const el = content();
  el.innerHTML = `
    <div class="page-header fade-in">
      <h1 class="page-title">My Taste</h1>
      <p class="page-subtitle">Build your personal profile. Select up to 5 games you love.</p>
    </div>
    <div class="search-section fade-in">
      <div id="taste-search"></div>
    </div>
    <div class="taste-grid fade-in" id="taste-grid"></div>
    <div class="taste-actions fade-in">
      <button class="btn btn-primary" id="taste-analyze" disabled>Analyze Profile</button>
      <span class="taste-msg" id="taste-msg">Select at least 1 game</span>
    </div>
    <div id="taste-results"></div>`;
  const sb = renderSearchBox('taste-input','Search to add a game...', obj=>{
    if(state.tasteGames.length>=5||state.tasteGames.some(g=>g.name===obj.name)) return;
    state.tasteGames.push(obj); renderTasteGrid(); $('#taste-input').value='';
  });
  $('#taste-search').appendChild(sb);
  renderTasteGrid();
  $('#taste-analyze').addEventListener('click', doTaste);
}
function renderTasteGrid(){
  const grid=$('#taste-grid'),btn=$('#taste-analyze'),msg=$('#taste-msg');
  if(!grid) return;
  if(!state.tasteGames.length){
    grid.innerHTML='<div class="taste-empty">No games selected yet.</div>';
    btn.disabled=true; msg.textContent='Select at least 1 game';
  } else {
    grid.innerHTML = state.tasteGames.map((g,i)=>`
      <div class="taste-card">
        <img class="taste-card-img" src="${g.img}" alt="${escHtml(g.name)}" loading="lazy">
        <div class="taste-card-overlay">
          <div class="taste-card-name">${escHtml(g.name)}</div>
        </div>
        <button class="taste-card-remove" data-idx="${i}">&times;</button>
      </div>`).join('');
    const rem = 5-state.tasteGames.length;
    for(let i=0;i<rem;i++) grid.innerHTML += '<div class="taste-card empty-slot"></div>';
    btn.disabled=false; msg.textContent=`${state.tasteGames.length}/5 selected`;
  }
  grid.querySelectorAll('.taste-card-remove').forEach(b=>b.addEventListener('click',()=>{ state.tasteGames.splice(+b.dataset.idx,1); renderTasteGrid(); }));
}
async function doTaste(){
  if(!state.tasteGames.length) return;
  const r=$('#taste-results'); r.innerHTML=skeletonList(6);
  try {
    const d = await api('/api/recommend/multi',{method:'POST',body:JSON.stringify({games:state.tasteGames.map(g=>g.name),count:12})});
    if(!d.results||!d.results.length){ r.innerHTML=`<div class="empty-state">Not enough data</div>`; return; }
    r.innerHTML=`<div class="section-title" style="margin-top:16px;">${d.results.length} matches from your favorites</div>`;
    const list=document.createElement('div'); list.className='rec-list stagger';
    d.results.forEach((g,i)=>list.appendChild(renderRecCard(g,i+1))); r.appendChild(list);
  } catch(e){ r.innerHTML=`<div class="status-msg error">Something went wrong.</div>`; }
}

// ── PLAYTIME ──
async function renderPlaytime(){
  const brackets = [{key:'short',title:'Quick',range:'Under 10h'},{key:'medium',title:'Standard',range:'10–30h'},{key:'long',title:'Deep Dive',range:'30–100h'},{key:'endless',title:'Lifestyle',range:'100h+'}];
  const el = content();
  el.innerHTML = `
    <div class="page-header fade-in">
      <h1 class="page-title">Playtime</h1>
      <p class="page-subtitle">Games that match your available time.</p>
    </div>
    <div class="bracket-grid fade-in" id="bracket-grid"></div>
    <div id="playtime-results"></div>`;
  const grid=$('#bracket-grid');
  brackets.forEach(b=>{
    const card=document.createElement('div'); card.className='bracket-card'+(state.selectedBracket===b.key?' selected':'');
    card.innerHTML=`<div class="bracket-card-title">${b.title}</div><div class="bracket-card-range">${b.range}</div>`;
    card.addEventListener('click',()=>{ state.selectedBracket=b.key; $$('.bracket-card',grid).forEach(c=>c.classList.remove('selected')); card.classList.add('selected'); loadPlaytime(b.key); });
    grid.appendChild(card);
  });
  if(state.selectedBracket) loadPlaytime(state.selectedBracket);
}
async function loadPlaytime(key){
  const r=$('#playtime-results'); r.innerHTML=skeletonList(6);
  try {
    const d = await api(`/api/playtime/${key}`);
    if(!d.results||!d.results.length){ r.innerHTML=`<div class="empty-state">No games found.</div>`; return; }
    r.innerHTML='';
    const list=document.createElement('div'); list.className='rec-list stagger';
    d.results.forEach((g,i)=>list.appendChild(renderRecCard(g,i+1))); r.appendChild(list);
  } catch(e){ r.innerHTML=`<div class="status-msg error">Failed to load.</div>`; }
}

// ── DEEP DIVE ──
async function renderExplore(gameName){
  const el = content();
  el.innerHTML = `
    <div class="page-header fade-in">
      <h1 class="page-title">Deep Dive</h1>
      <p class="page-subtitle">Explore a game and discover its closest neighbors.</p>
    </div>
    <div class="search-section fade-in">
      <div id="explore-search"></div>
    </div>
    <div id="explore-detail"></div>`;
  const sb = renderSearchBox('explore-input','Search for a game...', obj=>loadExplore(obj.name));
  $('#explore-search').appendChild(sb);
  if(gameName){ $('#explore-input').value=gameName; loadExplore(gameName); }
}

async function loadExplore(name){
  const detail=$('#explore-detail');
  detail.innerHTML=`<div class="skeleton" style="height:300px;margin-bottom:32px"></div><div style="display:grid;grid-template-columns:1fr 300px;gap:64px"><div><div class="skeleton skeleton-line w-60" style="height:32px;margin-bottom:16px"></div><div class="skeleton" style="height:200px"></div></div><div><div class="skeleton" style="height:300px"></div></div></div>`;

  try {
    const data = await api(`/api/explore/${encodeURIComponent(name)}`);
    const g = data.game, similar = data.similar||[];
    const genres=(g.genres||'').split(',').filter(Boolean).map(s=>s.trim());
    const year=(g.release_date||'').match(/\d{4}/)?.[0]||'';
    const totalRatings=g.positive_ratings+g.negative_ratings;
    const steamLink=g.steam_link||`https://store.steampowered.com/app/${g.appid}`;
    const hasTrailer=g.movies&&g.movies.length>0;
    const desc=g.about_the_game||g.description||'';
    const tags=(g.steamspy_tags||g.tags||'').split(',').filter(Boolean).map(s=>s.trim());
    const cats=(g.categories||'').split(',').filter(Boolean).map(s=>s.trim());
    const platforms=(g.platforms||'').split(';').map(s=>s.trim().toLowerCase());

    const highResHero = imgSrc(g.header_image).replace('header.jpg', 'library_hero.jpg');
    let heroHtml=`<div class="game-hero-embed"><img class="game-hero-embed-img" src="${highResHero}" onerror="this.onerror=null;this.src='${imgSrc(g.header_image)}';" alt="${escHtml(g.name)}"></div>`;

    const genreChips=genres.slice(0,4).map(ge=>`<span class="tag tag-filled">${escHtml(ge)}</span>`).join('');

    let trHtml='';
    if(hasTrailer){
      trHtml=`
      <div class="detail-section">
        <h3 class="detail-section-title">Trailer</h3>
        <div class="video-embed-container">
          <video style="width: 100%; height: 100%; position: absolute; top: 0; left: 0; background:#000;" controls preload="none" poster="${imgSrc(g.header_image)}">
            <source src="${httpsUrl(g.movies[0])}" type="video/webm">
            <source src="${httpsUrl(g.movies[0]).replace('.webm','.mp4')}" type="video/mp4">
          </video>
        </div>
      </div>`;
    }

    let ssHtml='';
    if(g.screenshots&&g.screenshots.length){
      ssHtml=`
      <div class="detail-section">
        <h3 class="detail-section-title">Screenshots</h3>
        <div class="screenshots-row">
          ${g.screenshots.map(u=>`<img class="screenshot-thumb" src="${httpsUrl(u)}" alt="Screenshot" loading="lazy" onclick="openLightbox('${httpsUrl(u)}')">`).join('')}
        </div>
      </div>`;
    }

    let descHtml='';
    if(desc){
      const isLong=desc.length>600;
      descHtml=`
      <div class="detail-section">
        <h3 class="detail-section-title">About This Game</h3>
        <div class="game-description-box ${isLong?'truncated':''}" id="desc-box">${desc}</div>
        ${isLong?'<button class="btn btn-ghost expand-btn" id="expand-desc">Read More</button>':''}
      </div>`;
    }

    let simHtml='';
    if(similar.length){
      simHtml=`
      <div class="detail-section">
        <h3 class="detail-section-title">Similar Games</h3>
        <div class="similar-grid stagger">
          ${similar.map(s=>{
            const sg=(s.genres||'').split(',').slice(0,2).join(', ');
            return `
            <div class="similar-card" onclick="navigateToExplore('${escHtml(s.name).replace(/'/g,"\\'")}')">
              <img class="similar-card-img" src="${imgSrc(s.header_image)}" alt="${escHtml(s.name)}" loading="lazy">
              <div class="similar-card-body">
                <div class="similar-card-name">${escHtml(s.name)}</div>
                <div class="similar-card-meta">${escHtml(sg)}</div>
                <div class="similar-card-score">${s.similarity}% Match</div>
              </div>
            </div>`;
          }).join('')}
        </div>
      </div>`;
    }

    const tagsHtml=tags.length?tags.slice(0,10).map(t=>`<span class="tag tag-outline">${escHtml(t)}</span>`).join(''):'';
    const catsHtml=cats.length?cats.slice(0,8).map(c=>`<span class="tag tag-outline">${escHtml(c)}</span>`).join(''):'';

    detail.innerHTML=`
    <div class="explore-full fade-in">
      ${heroHtml}
      
      <div class="explore-content-grid">
        <div class="explore-main">
          <div class="explore-header">
            <h1 class="explore-title">${escHtml(g.name)}</h1>
            <div class="explore-meta-inline">
              <span>${escHtml(g.developer||'Unknown Developer')}</span>
              ${year?`<span class="meta-sep">/</span><span>${year}</span>`:''}
              ${g.price>0?`<span class="meta-sep">/</span><span>$${g.price}</span>`:`<span class="meta-sep">/</span><span class="free-badge">Free</span>`}
            </div>
            ${genreChips?`<div class="genre-row">${genreChips}</div>`:''}
          </div>
          
          ${descHtml}
          ${trHtml}
          ${ssHtml}
          ${simHtml}
        </div>
        
        <aside class="explore-sidebar">
          <div class="panel">
            <h3 class="panel-title">Details</h3>
            <div class="meta-list">
              <div class="meta-item"><span class="meta-label">Release Date</span><span class="meta-value">${escHtml(g.release_date||'TBD')}</span></div>
              <div class="meta-item"><span class="meta-label">Developer</span><span class="meta-value">${escHtml(g.developer||'-')}</span></div>
              <div class="meta-item"><span class="meta-label">Publisher</span><span class="meta-value">${escHtml(g.publisher||g.developer||'-')}</span></div>
              <div class="meta-item"><span class="meta-label">Price</span><span class="meta-value">${g.price>0?'$'+g.price:'Free'}</span></div>
              ${g.metacritic_score>0?`<div class="meta-item"><span class="meta-label">Metacritic</span><span class="meta-value highlight-score">${g.metacritic_score}</span></div>`:''}
            </div>
          </div>
          
          ${totalRatings>0?`
          <div class="panel">
            <h3 class="panel-title">Community Reviews</h3>
            <div class="review-block">
              <div class="review-pct">${g.quality_score}%</div>
              <div class="review-text">Positive<br><span class="review-sub">${formatNum(totalRatings)} ratings</span></div>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="width:${g.quality_score}%"></div></div>
          </div>`:''}
          
          <a class="btn btn-primary w-full" href="${steamLink}" target="_blank" rel="noopener" style="justify-content:center; margin-bottom: 24px;">View on Steam ↗</a>
          
          ${catsHtml||tagsHtml?`
          <div class="panel">
            ${catsHtml?`<h3 class="panel-title">Features</h3><div class="tag-cloud" style="margin-bottom:20px;">${catsHtml}</div>`:''}
            ${tagsHtml?`<h3 class="panel-title">Tags</h3><div class="tag-cloud">${tagsHtml}</div>`:''}
          </div>`:''}
        </aside>
      </div>
    </div>`;

    const eb=$('#expand-desc'); if(eb) eb.addEventListener('click',()=>{ $('#desc-box').classList.remove('truncated'); eb.remove(); });
  } catch(e){ detail.innerHTML=`<div class="status-msg error">Could not load game details.</div>`; console.error(e); }
}

// ── TRENDING ──
async function renderTrending(){
  const el = content();
  el.innerHTML = `
    <div class="page-header fade-in">
      <h1 class="page-title">Trending</h1>
      <p class="page-subtitle">Games ranked by community engagement.</p>
    </div>
    <div class="chart-container fade-in"><canvas id="trending-chart"></canvas></div>
    <div id="trending-table" class="fade-in">${skeletonList(5)}</div>`;
  try {
    const data = await api('/api/trending?limit=15');
    const results = data.results||[];
    if(typeof Chart!=='undefined'&&results.length){
      const ctx=document.getElementById('trending-chart');
      if(ctx){
        const isDark=document.documentElement.getAttribute('data-theme')==='dark';
        const color = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#111';
        const gridColor = isDark ? '#222' : '#eee';
        const textColor = isDark ? '#888' : '#777';
        
        new Chart(ctx,{
          type:'bar',
          data:{
            labels:results.map(r=>r.name.length>22?r.name.slice(0,22)+'…':r.name),
            datasets:[{
              label:'Total Hours',
              data:results.map(r=>r.total_hours),
              backgroundColor:color,
              maxBarThickness:28
            }]
          },
          options:{
            indexAxis:'y',
            responsive:true,
            maintainAspectRatio:false,
            plugins:{legend:{display:false}},
            scales:{
              x:{grid:{color:gridColor},ticks:{color:textColor,font:{family:'Inter',size:12}}},
              y:{grid:{display:false},ticks:{color:textColor,font:{family:'Inter',size:12}}}
            }
          }
        });
        ctx.parentElement.style.height=Math.max(300,results.length*36)+'px';
      }
    }
    
    $('#trending-table').innerHTML=`
      <table class="data-table">
        <thead>
          <tr>
            <th style="width: 48px">Rank</th>
            <th>Game</th>
            <th class="text-right">Total Hours</th>
            <th class="text-right">Players</th>
          </tr>
        </thead>
        <tbody>
          ${results.map((r,i)=>`
          <tr>
            <td class="rank-cell">${i+1}</td>
            <td>
              <div class="table-game-cell">
                ${r.header_image?`<img class="table-thumb" src="${imgSrc(r.header_image)}" alt="" loading="lazy">`:''}
                <span class="table-game-name">${escHtml(r.name)}</span>
              </div>
            </td>
            <td class="text-right num-cell">${formatNum(r.total_hours)}</td>
            <td class="text-right num-cell">${formatNum(r.n_players)}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch(e){ console.error(e); }
}

// ═══════════════════════════════════════════
// LIGHTBOX & NAVIGATION
// ═══════════════════════════════════════════

function openLightbox(url){ const lb=$('#lightbox'); $('#lightbox-img').src=url; lb.classList.add('open'); }
window.openLightbox=openLightbox;
function navigateToExplore(name){ window.location.hash='#explore'; setTimeout(()=>renderExplore(name),50); }
window.navigateToExplore=navigateToExplore;

function route(){
  const hash=(window.location.hash||'#home').slice(1);
  const page=pages[hash]?hash:'home';
  state.page=page;
  $$('.nav-link').forEach(l=>l.classList.toggle('active',l.dataset.page===page));
  pages[page]();
  const topNav = $('#top-nav');
  if(topNav.classList.contains('mobile-open')) topNav.classList.remove('mobile-open');
}

const pages = { home:renderHome, discover:renderDiscover, mood:renderMood, taste:renderTaste, playtime:renderPlaytime, explore:renderExplore, trending:renderTrending };

document.addEventListener('DOMContentLoaded',()=>{
  initTheme();
  $('#theme-toggle').addEventListener('click', toggleTheme);
  
  $('#mobile-menu-btn').addEventListener('click', ()=>{
    $('#top-nav').classList.toggle('mobile-open');
  });
  
  $('#lightbox-close').addEventListener('click',()=>$('#lightbox').classList.remove('open'));
  $('#lightbox').addEventListener('click',e=>{ if(e.target===e.currentTarget) $('#lightbox').classList.remove('open'); });
  
  route();
});
window.addEventListener('hashchange', route);

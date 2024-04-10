import{c as h,r as x,x as t,E as o,C as l}from"./index-rBgYIjBV.js";/**
 * @license lucide-react v0.319.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const m=h("Clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]),a=window.location.hostname,n=`<iframe 
src="https://${a}/open-kf-chatbot"
title="Chatbot"
style="min-width: 420px;min-height: 60vh"
frameborder="0"
></iframe>`,d=`<script 
src="https://${a}/open-kf-chatbot/embed.js" 
bot-domain="https://${a}/open-kf-chatbot" 
defer
><\/script>`,f=()=>{const[c,i]=x.useState({script:!1,iframe:!1}),r=e=>{navigator.clipboard.writeText(e==="script"?d:n),i(s=>({...s,[e]:!0})),setTimeout(()=>{i(s=>({...s,[e]:!1}))},1500)};return t.jsxs("div",{className:"flex flex-col items-center",children:[t.jsx("div",{className:"mt-2",children:t.jsx("p",{className:"text-sm text-zinc-500",children:"To add a chat bubble to the bottom right of your website add this script tag to your html"})}),t.jsx("div",{className:"mt-5",children:t.jsx("pre",{className:"w-fit overflow-auto rounded bg-zinc-100 p-2 text-xs",children:t.jsx("code",{children:d})})}),t.jsxs(o,{className:"mt-6",variant:"outline",onClick:()=>r("script"),children:["Copy Script",c.script?t.jsx(l,{className:"h-4 w-4 ml-2"}):t.jsx(m,{className:"h-4 w-4 ml-2"})]}),t.jsx("div",{className:"mt-2",children:t.jsx("p",{className:"text-sm text-zinc-500",children:"To add the chatbot any where on your website, add this iframe to your html code"})}),t.jsx("div",{className:"mt-5",children:t.jsx("pre",{className:"w-fit overflow-auto rounded bg-zinc-100 p-2 text-xs",children:t.jsx("code",{children:n})})}),t.jsxs(o,{className:"mt-6",variant:"outline",onClick:()=>r("iframe"),children:["Copy Iframe",c.iframe?t.jsx(l,{className:"h-4 w-4 ml-2"}):t.jsx(m,{className:"h-4 w-4 ml-2"})]})]})};export{f as Embed};

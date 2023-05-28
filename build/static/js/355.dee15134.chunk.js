"use strict";(self.webpackChunkbartender_frontend=self.webpackChunkbartender_frontend||[]).push([[355],{3355:function(t,n,e){e.r(n),e.d(n,{default:function(){return tt}});var r=e(4165),o=e(5861),s=e(9439),i=e(168),a=e(1413),l=e(6573),g=[500,400,551,560,800],c=l.Z.create({timeout:12e4});c.interceptors.request.use((function(t){return"get"===t.method&&(t.data={}),t}),(function(t){return console.warn(t),Promise.reject(t)})),c.interceptors.response.use((function(t){var n=t.data,e=n.code,r=n.message;return g.includes(e)&&console.error(r),n}),(function(t){return Promise.reject(t)}));var u,d,p,h,x,A=c,f=function(){var t=(0,o.Z)((0,r.Z)().mark((function t(n){var e;return(0,r.Z)().wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,A({url:"/v1/chat",data:(0,a.Z)({},n),method:"post"});case 2:return e=t.sent,t.abrupt("return",e);case 4:case"end":return t.stop()}}),t)})));return function(n){return t.apply(this,arguments)}}(),m=e(1961),Z=e(8792),B=e(5045),C=e(5386),k=e(6417),Q=(0,m.zo)("div")(u||(u=(0,i.Z)(["\n\tpadding-top: 120px;\n\tflex: 1;\n\tdisplay: flex;\n\tflex-direction: column;\n"]))),v=(0,m.zo)("h2")(d||(d=(0,i.Z)(["\n\tfont-size: 38px;\n\tcolor: #333;\n\tfont-weight: 600;\n\tdisplay: flex;\n\talign-items: center;\n\tflex-wrap: wrap;\n\tfont-family: PingFang SC-Semibold, PingFang SC;\n\timg {\n\t\twidth: 60px;\n\t}\n"]))),w=(0,m.zo)("div")(p||(p=(0,i.Z)(["\n\tfont-size: 16px;\n\tcolor: #26334b;\n\tmargin-top: 4px;\n\tspan {\n\t\tpadding: 2px 8px;\n\t\tborder: 1px solid #26334b;\n\t\tborder-radius: 4px;\n\t}\n\ta {\n\t\tcolor: #1677ff;\n\t}\n"]))),b=(0,m.zo)("div")(h||(h=(0,i.Z)(["\n\tmargin-top: 50px;\n\tdisplay: flex;\n\talign-items: center;\n\tjustify-content: space-between;\n\tgap: 12px;\n\tflex-flow: wrap;\n"]))),E=(0,m.zo)("div")(x||(x=(0,i.Z)(["\n\tmin-width: 212px;\n\tmin-height: 138px;\n\tflex: 1;\n\tpadding: 16px;\n\tborder-radius: 12px;\n\tbackground-color: #f5f5f5;\n\timg {\n\t\twidth: 24px;\n\t\tmargin-bottom: 12px;\n\t}\n\th3 {\n\t\tfont-size: 16px;\n\t\tmargin-bottom: 5px;\n\t\tcolor: #26334b;\n\t\tfont-weight: 700;\n\t}\n\tdiv {\n\t\tfont-size: 12px;\n\t\tcolor: #26334b;\n\t}\n"])));var S,I,T,J,W,y,z,D,Y=function(){return(0,k.jsxs)(Q,{children:[(0,k.jsxs)(v,{children:[(0,k.jsx)("img",{src:"data:image/png;base64,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",alt:""}),"Bartender NPC MVP for Gaggle Studio"]}),(0,k.jsx)(w,{children:'We noticed that sometimes the bartender will not output a response upon entering a prompt. We suggest you "clear" again, or wait a few minutes before trying again."'}),(0,k.jsxs)(w,{style:{marginTop:"10px"},children:["Please reference"," ",(0,k.jsx)("a",{href:"https://docs.google.com/document/d/1-odQBPK6H4IpT_ew5HAZgke4mhWsGJvi5d0SIHznK5Y/edit",target:"_blank",rel:"noreferrer",children:"this link"})," ","for capabilities/limitations and general overview"]}),(0,k.jsx)(b,{children:[{id:"cuanxieduanwen",icon:"https://www.imageoss.com/images/2023/04/23/Frame2x132f6276a56cf44e81.png",name:"Instructions and Commands",descList:["Please clear memory before you start.",'Type "toggleThoughts" to turn on/off thought prompts','Type "clear" to clear memory',"Please clear memory after you are done with the session.","Type \"viewMemory: {'username'}\" to view past conversation history\n\t\t\t\twith a user"]}].map((function(t){return(0,k.jsxs)(E,{children:[(0,k.jsx)("img",{src:t.icon,alt:""}),(0,k.jsx)("h3",{children:t.name}),(0,k.jsx)("div",{children:t.descList.map((function(t,n){return(0,k.jsx)("p",{children:"".concat(n+1,". ").concat(t)})}))})]},t.id)}))})]})},F=e(5554),j=e(7313),R=e(816),U=e.n(R),K=e(8629),M=e.n(K),q=(0,m.zo)("div")(S||(S=(0,i.Z)(["\n\tdisplay: flex;\n\tmargin-top: 12px;\n"]))),V=(0,m.zo)("img")(I||(I=(0,i.Z)(["\n\twidth: 34px;\n\theight: 34px;\n\tborder-radius: 50%;\n"]))),L=(0,m.zo)("div")(T||(T=(0,i.Z)(["\n\tdisplay: flex;\n\tflex-direction: column;\n\twidth: calc(100% - 42px);\n"]))),G=(0,m.zo)("span")(J||(J=(0,i.Z)(["\n\tmargin-top: 4px;\n\tmargin-bottom: 4px;\n\tfont-size: 12px;\n\tcolor: #999;\n"]))),H=(0,m.zo)("div")(W||(W=(0,i.Z)(["\n\tmax-width: calc(100% - 42px);\n\tdisplay: inline-block;\n\tpadding: 12px;\n\n\t&.left {\n\t\tmargin-right: auto;\n\t\tbackground-color: #f4f6f8;\n\t\tborder-top-left-radius: 2px;\n\t\tborder-top-right-radius: 12px;\n\t\tborder-bottom-left-radius: 12px;\n\t\tborder-bottom-right-radius: 12px;\n\t}\n\t&.right {\n\t\tmargin-left: auto;\n\t\tbackground-color: #cdeeff;\n\t\tborder-top-left-radius: 12px;\n\t\tborder-top-right-radius: 2px;\n\t\tborder-bottom-left-radius: 12px;\n\t\tborder-bottom-right-radius: 12px;\n\t}\n"]))),O=function(t){var n=t.position,e=t.text,r=t.time,o=t.status,s=t.scrollFn,i=(0,F.v9)((function(t){return{messages:t.chats.messages}})).messages,l=(0,F.I0)();return(0,k.jsxs)(q,{style:{justifyContent:"right"===n?"flex-end":"flex-start"},children:["left"===n&&(0,k.jsx)(V,{style:{marginRight:8},src:"https://cdn.jsdelivr.net/gh/duogongneng/testuitc/svg-1681898659579.svg"}),(0,k.jsxs)(L,{children:[(0,k.jsx)(G,{style:{textAlign:"right"===n?"right":"left"},children:r}),(0,k.jsx)(H,{className:"right"===n?"right":"left",children:"left"===n&&"loading"===o?(0,k.jsx)(M(),{cursor:{hideWhenDone:!0},onTypingDone:function(){var t=(0,a.Z)((0,a.Z)({},i[i.length-1]),{},{status:"done"});l({type:"UPDATE_CHAT",value:(0,a.Z)({},t)}),s()},children:e}):(0,k.jsx)("div",{style:{whiteSpace:"pre-wrap"},dangerouslySetInnerHTML:{__html:e.replace(/\n/g,"<br>")}})})]}),"right"===n&&(0,k.jsx)(V,{style:{marginLeft:8},src:"https://cdn.jsdelivr.net/gh/duogongneng/testuitc/1682426702646avatarf3db669b024fad66-1930929abe2847093.png"})]})},P=(0,j.memo)(O),N=Z.Z.TextArea,X=(0,m.zo)("div")(y||(y=(0,i.Z)(["\n\tdisplay: flex;\n\tflex-direction: column;\n\talign-items: center;\n\tjustify-content: center;\n\tgap: 10px;\n\tmax-width: 900px;\n\theight: calc(100vh - 56px);\n\tpadding: 10px 20px;\n\tmargin: 0 auto;\n"]))),_=(0,m.zo)("div")(z||(z=(0,i.Z)(["\n\tflex: 1;\n\twidth: 100%;\n\tmax-width: 900px;\n\talign-self: center;\n\toverflow: auto;\n\t&::-webkit-scrollbar {\n\t\tdisplay: none;\n\t}\n"]))),$=(0,m.zo)("div")(D||(D=(0,i.Z)(["\n\twidth: 100%;\n\tdisplay: flex;\n\tjustify-content: center;\n\talign-items: center;\n\tgap: 10px;\n"]))),tt=function(){var t=(0,j.useRef)(null),n=function(t){var n=function(){var n=(0,o.Z)((0,r.Z)().mark((function n(){return(0,r.Z)().wrap((function(n){for(;;)switch(n.prev=n.next){case 0:t&&(t.scrollTop=t.scrollHeight);case 1:case"end":return n.stop()}}),n)})));return function(){return n.apply(this,arguments)}}(),e=function(){var n=(0,o.Z)((0,r.Z)().mark((function n(){return(0,r.Z)().wrap((function(n){for(;;)switch(n.prev=n.next){case 0:t&&(t.scrollTop=0);case 1:case"end":return n.stop()}}),n)})));return function(){return n.apply(this,arguments)}}(),s=function(){var n=(0,o.Z)((0,r.Z)().mark((function n(){return(0,r.Z)().wrap((function(n){for(;;)switch(n.prev=n.next){case 0:t&&t.scrollHeight-t.scrollTop-t.clientHeight<=100&&(t.scrollTop=t.scrollHeight);case 1:case"end":return n.stop()}}),n)})));return function(){return n.apply(this,arguments)}}();return{scrollElement:t,scrollToBottom:n,scrollToTop:e,scrollToBottomIfAtBottom:s}}(t.current),e=n.scrollToBottomIfAtBottom,i=n.scrollToBottom,a=(0,F.v9)((function(t){return{messages:t.chats.messages}})).messages,l=(0,F.I0)(),g=(0,j.useState)(""),c=(0,s.Z)(g,2),u=c[0],d=c[1],p=(0,j.useState)(!1),h=(0,s.Z)(p,2),x=h[0],A=h[1];console.log("messages:",a);var m=(0,j.useCallback)((0,o.Z)((0,r.Z)().mark((function t(){var n,e,o,s;return(0,r.Z)().wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return A(!0),n=U()().format("YYYY-MM-DD HH:mm:ss"),l({type:"ADD_CHAT",value:{time:n,content:u,role:"user",status:"done"}}),t.prev=4,t.next=7,f({user_input:u});case 7:(e=t.sent)&&e.result&&(o=U()().format("YYYY-MM-DD HH:mm:ss"),s={time:o,content:e.result,role:"assistant",status:"loading"},l({type:"ADD_CHAT",value:s}),A(!1)),t.next=15;break;case 11:t.prev=11,t.t0=t.catch(4),console.log("error:",t.t0),A(!1);case 15:case"end":return t.stop()}}),t,null,[[4,11]])}))),[l,u]);return(0,j.useLayoutEffect)((function(){t&&i()}),[t,a,i]),(0,k.jsxs)(X,{children:[(0,k.jsxs)(_,{ref:t,children:[a.map((function(t,n){return(0,k.jsx)(P,{position:"user"===t.role?"right":"left",text:t.content,time:t.time,status:t.status,scrollFn:i},n)})),a.length<=0&&(0,k.jsx)(Y,{})]}),(0,k.jsxs)($,{children:[(0,k.jsx)(B.ZP,{size:"large",icon:(0,k.jsx)(C.Z,{}),onClick:function(){l({type:"CLEAR_CHAT"})}}),(0,k.jsx)(N,{size:"large",value:u,placeholder:"Say something...",autoSize:{minRows:1,maxRows:5},style:{width:"100%"},onPressEnter:function(t){"Enter"===t.key&&13===t.keyCode&&t.shiftKey||"Enter"===t.key&&13===t.keyCode&&(u.length>0&&(m(),e(),d("")),t.preventDefault())},onChange:function(t){d(t.target.value)}}),(0,k.jsx)(B.ZP,{type:"primary",size:"large",loading:x,onClick:function(){m(),e(),d("")},children:x?"Generating":"Send"})]})]})}}}]);
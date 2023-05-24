"use strict";(self.webpackChunkbartender_frontend=self.webpackChunkbartender_frontend||[]).push([[638],{5394:function(t,n,e){e.r(n),e.d(n,{default:function(){return Q}});var r=e(4165),o=e(5861),i=e(9439),s=e(168),a=e(1413),c=e(6573),l=[500,400,551,560,800],p=c.Z.create({timeout:12e4});p.interceptors.request.use((function(t){return t.headers.Authorization="Bearer ".concat("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"),"get"===t.method&&(t.data={}),t}),(function(t){return console.warn(t),Promise.reject(t)})),p.interceptors.response.use((function(t){var n=t.data,e=n.code,r=n.message;return l.includes(e)&&console.error(r),n}),(function(t){return Promise.reject(t)}));var u,d,f,x,h,g=p,m=function(){var t=(0,o.Z)((0,r.Z)().mark((function t(n){var e;return(0,r.Z)().wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,g({url:"/v1/chat",data:(0,a.Z)({},n),method:"post"});case 2:return e=t.sent,t.abrupt("return",e);case 4:case"end":return t.stop()}}),t)})));return function(n){return t.apply(this,arguments)}}(),v=e(1961),b=e(8792),w=e(5045),y=e(5386),j=e(6417),Z=(0,v.zo)("div")(u||(u=(0,s.Z)(["\n\tpadding-top: 120px;\n\tflex: 1;\n\tdisplay: flex;\n\tflex-direction: column;\n"]))),k=(0,v.zo)("h2")(d||(d=(0,s.Z)(["\n\tfont-size: 38px;\n\tcolor: #333;\n\tfont-weight: 600;\n\tdisplay: flex;\n\talign-items: center;\n\tfont-family: PingFang SC-Semibold, PingFang SC;\n\timg {\n\t\twidth: 60px;\n\t}\n"]))),z=(0,v.zo)("p")(f||(f=(0,s.Z)(["\n\tfont-size: 16px;\n\tcolor: #26334b;\n\tmargin-top: 4px;\n\tspan {\n\t\tpadding: 2px 8px;\n\t\tborder: 1px solid #26334b;\n\t\tborder-radius: 4px;\n\t}\n"]))),T=(0,v.zo)("div")(x||(x=(0,s.Z)(["\n\tmargin-top: 100px;\n\tdisplay: flex;\n\talign-items: center;\n\tjustify-content: space-between;\n\tgap: 12px;\n\tflex-flow: wrap;\n"]))),C=(0,v.zo)("div")(h||(h=(0,s.Z)(["\n\tmin-width: 212px;\n\tmin-height: 138px;\n\tflex: 1;\n\tpadding: 16px;\n\tborder-radius: 12px;\n\tbackground-color: #f5f5f5;\n\timg {\n\t\twidth: 24px;\n\t\tmargin-bottom: 12px;\n\t}\n\th3 {\n\t\tfont-size: 16px;\n\t\tmargin-bottom: 5px;\n\t\tcolor: #26334b;\n\t\tfont-weight: 700;\n\t}\n\tdiv {\n\t\tfont-size: 12px;\n\t\tcolor: #26334b;\n\t}\n"])));var A,H,D,S,B,Y,E,P,_=function(){var t=[{id:"cuanxieduanwen",icon:"https://www.imageoss.com/images/2023/04/23/Frame2x132f6276a56cf44e81.png",name:"Extra Commands",desc:(0,j.jsxs)(j.Fragment,{children:[(0,j.jsx)("p",{children:'1. Type "toggleThoughts" to turn on/off thought prompts'}),(0,j.jsx)("p",{children:'2. Type "clear" to clear memory'}),(0,j.jsxs)("p",{children:['3. Type "viewMemory: ',"username",'" to view past conversation history with a user']})]})}];return(0,j.jsxs)(Z,{children:[(0,j.jsxs)(k,{children:[(0,j.jsx)("img",{src:"https://www.imageoss.com/images/2023/04/23/robot-logo4987eb2ca3f5ec85.png",alt:""}),"Welcome to Bartender"]}),(0,j.jsx)(z,{children:"\u4e0eAI\u667a\u80fd\u804a\u5929\uff0c\u7545\u60f3\u65e0\u9650\u53ef\u80fd\uff01\u57fa\u4e8e\u5148\u8fdb\u7684AI\u5f15\u64ce\uff0c\u8ba9\u4f60\u7684\u4ea4\u6d41\u66f4\u52a0\u667a\u80fd\u3001\u9ad8\u6548\u3001\u4fbf\u6377\uff01"}),(0,j.jsxs)(z,{children:[(0,j.jsx)("span",{children:"Shift"})," + ",(0,j.jsx)("span",{children:"Enter"})," new line"]}),(0,j.jsx)(T,{children:t.map((function(t){return(0,j.jsxs)(C,{children:[(0,j.jsx)("img",{src:t.icon,alt:""}),(0,j.jsx)("h3",{children:t.name}),(0,j.jsx)("div",{children:t.desc})]},t.id)}))})]})},M=e(5554),R=e(7313),I=e(816),F=e.n(I),L=(0,v.zo)("div")(A||(A=(0,s.Z)(["\n\tdisplay: flex;\n\tmargin-top: 12px;\n"]))),G=(0,v.zo)("img")(H||(H=(0,s.Z)(["\n\twidth: 34px;\n\theight: 34px;\n\tborder-radius: 50%;\n"]))),q=(0,v.zo)("div")(D||(D=(0,s.Z)(["\n\tdisplay: flex;\n\tflex-direction: column;\n\twidth: calc(100% - 42px);\n"]))),J=(0,v.zo)("span")(S||(S=(0,s.Z)(["\n\tmargin-top: 4px;\n\tmargin-bottom: 4px;\n\tfont-size: 12px;\n\tcolor: #999;\n"]))),K=(0,v.zo)("div")(B||(B=(0,s.Z)(["\n\tmax-width: calc(100% - 42px);\n\tdisplay: inline-block;\n\tpadding: 12px;\n\n\t&.left {\n\t\tmargin-right: auto;\n\t\tbackground-color: #f4f6f8;\n\t\tborder-top-left-radius: 2px;\n\t\tborder-top-right-radius: 12px;\n\t\tborder-bottom-left-radius: 12px;\n\t\tborder-bottom-right-radius: 12px;\n\t}\n\t&.right {\n\t\tmargin-left: auto;\n\t\tbackground-color: #cdeeff;\n\t\tborder-top-left-radius: 12px;\n\t\tborder-top-right-radius: 2px;\n\t\tborder-bottom-left-radius: 12px;\n\t\tborder-bottom-right-radius: 12px;\n\t}\n"]))),N=function(t){var n=t.position,e=t.text,r=t.time;return(0,j.jsxs)(L,{style:{justifyContent:"right"===n?"flex-end":"flex-start"},children:["left"===n&&(0,j.jsx)(G,{style:{marginRight:8},src:"https://cdn.jsdelivr.net/gh/duogongneng/testuitc/svg-1681898659579.svg"}),(0,j.jsxs)(q,{children:[(0,j.jsx)(J,{style:{textAlign:"right"===n?"right":"left"},children:r}),(0,j.jsx)(K,{className:"right"===n?"right":"left",children:(0,j.jsx)("div",{dangerouslySetInnerHTML:{__html:e}})})]}),"right"===n&&(0,j.jsx)(G,{style:{marginLeft:8},src:"https://cdn.jsdelivr.net/gh/duogongneng/testuitc/1682426702646avatarf3db669b024fad66-1930929abe2847093.png"})]})},O=(0,R.memo)(N),U=b.Z.TextArea,V=(0,v.zo)("div")(Y||(Y=(0,s.Z)(["\n\tdisplay: flex;\n\tflex-direction: column;\n\talign-items: center;\n\tjustify-content: center;\n\tgap: 10px;\n\tmax-width: 900px;\n\theight: calc(100vh - 56px);\n\tpadding: '10px 20px';\n\tmargin: 0 auto;\n"]))),W=(0,v.zo)("div")(E||(E=(0,s.Z)(["\n\tflex: 1;\n\twidth: 100%;\n\tmax-width: 900px;\n\talign-self: center;\n\toverflow: auto;\n\t&::-webkit-scrollbar {\n\t\tdisplay: none;\n\t}\n"]))),X=(0,v.zo)("div")(P||(P=(0,s.Z)(["\n\twidth: 100%;\n\tdisplay: flex;\n\tjustify-content: center;\n\talign-items: center;\n\tgap: 10px;\n"]))),Q=function(){var t=(0,R.useRef)(null),n=function(t){var n=function(){var n=(0,o.Z)((0,r.Z)().mark((function n(){return(0,r.Z)().wrap((function(n){for(;;)switch(n.prev=n.next){case 0:t&&(t.scrollTop=t.scrollHeight);case 1:case"end":return n.stop()}}),n)})));return function(){return n.apply(this,arguments)}}(),e=function(){var n=(0,o.Z)((0,r.Z)().mark((function n(){return(0,r.Z)().wrap((function(n){for(;;)switch(n.prev=n.next){case 0:t&&(t.scrollTop=0);case 1:case"end":return n.stop()}}),n)})));return function(){return n.apply(this,arguments)}}(),i=function(){var n=(0,o.Z)((0,r.Z)().mark((function n(){return(0,r.Z)().wrap((function(n){for(;;)switch(n.prev=n.next){case 0:t&&t.scrollHeight-t.scrollTop-t.clientHeight<=100&&(t.scrollTop=t.scrollHeight);case 1:case"end":return n.stop()}}),n)})));return function(){return n.apply(this,arguments)}}();return{scrollElement:t,scrollToBottom:n,scrollToTop:e,scrollToBottomIfAtBottom:i}}(t.current),e=n.scrollToBottomIfAtBottom,s=n.scrollToBottom,a=(0,M.v9)((function(t){return{messages:t.chats.messages}})).messages,c=(0,M.I0)(),l=(0,R.useState)(""),p=(0,i.Z)(l,2),u=p[0],d=p[1],f=(0,R.useState)(!1),x=(0,i.Z)(f,2),h=x[0],g=x[1];console.log("messages:",a);var v=(0,R.useCallback)((0,o.Z)((0,r.Z)().mark((function t(){var n,e,o,i;return(0,r.Z)().wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return g(!0),n=F()().format("YYYY-MM-DD HH:mm:ss"),c({type:"ADD_CHAT",value:{time:n,content:u,role:"user"}}),t.prev=4,t.next=7,m({user_input:u});case 7:(e=t.sent)&&e.result&&(o=F()().format("YYYY-MM-DD HH:mm:ss"),i={time:o,content:e.result,role:"assistant"},c({type:"ADD_CHAT",value:i}),g(!1)),t.next=15;break;case 11:t.prev=11,t.t0=t.catch(4),console.log("error:",t.t0),g(!1);case 15:case"end":return t.stop()}}),t,null,[[4,11]])}))),[c,u]);return(0,R.useLayoutEffect)((function(){t&&s()}),[t,a,s]),(0,j.jsxs)(V,{children:[(0,j.jsxs)(W,{ref:t,children:[a.map((function(t,n){return(0,j.jsx)(O,{position:"user"===t.role?"right":"left",text:t.content,time:t.time},n)})),a.length<=0&&(0,j.jsx)(_,{})]}),(0,j.jsxs)(X,{children:[(0,j.jsx)(w.ZP,{size:"large",icon:(0,j.jsx)(y.Z,{}),onClick:function(){c({type:"CLEAR_CHAT"})}}),(0,j.jsx)(U,{size:"large",value:u,placeholder:"Say something...",autoSize:{minRows:1,maxRows:5},style:{width:"100%"},onPressEnter:function(t){"Enter"===t.key&&13===t.keyCode&&t.shiftKey||"Enter"===t.key&&13===t.keyCode&&(u.length>0&&(v(),e(),d("")),t.preventDefault())},onChange:function(t){d(t.target.value)}}),(0,j.jsx)(w.ZP,{type:"primary",size:"large",loading:h,onClick:function(){v(),e(),d("")},children:h?"Generating":"Send"})]})]})}}}]);
-- How the manual's marked names print: a control ({.guilabel}) and a
-- channel drawer ({.channel}) bold, a setting ({.setting}) in code type.
function Span(el)
  if el.classes:includes('guilabel') or el.classes:includes('channel') then
    return pandoc.Strong(el.content)
  elseif el.classes:includes('setting') then
    return pandoc.Code(pandoc.utils.stringify(el.content))
  end
end

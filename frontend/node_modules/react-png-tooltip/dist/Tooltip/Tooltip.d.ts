import * as React from 'react';
interface ITooltipProps {
    tooltip: React.ReactNode;
    fill: string;
    background: string;
    wrapperClassName: string;
    className: string;
    timeoutDelay: number;
    shouldDisableHover: boolean;
    shouldDisableClick: boolean;
    children: React.ReactNode;
}
declare const tooltip: (props: ITooltipProps) => JSX.Element;
export default tooltip;
